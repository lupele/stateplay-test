"""Synchronous rollout -> scoring -> forward NFT, with optimizer-boundary recovery."""
import argparse
from contextlib import nullcontext
import datetime
import math
import os
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from safetensors.torch import load_file
from .config import load_config,validate
from .io import run_path,read_json,write_json,append_jsonl,lock,digest,reward_path,publish_link,reward_checkpoint,world_checkpoint
from .data.conditions import catalog,prepare_group,complete_protocol
from .data.sampler import Ledger
from .adapters.world_model import WorldAdapter
from .adapters.parameter_modes import optimizer,gradient_norms,parameter_hash
from .adapters.checkpoint import save,load,rng_state
from .algorithms.policy_bank import PolicyBank
from .algorithms.nft_loss import forward_noise,nft_loss
from .algorithms.reward_transform import transform
from .algorithms.anchors import AnchorScheduler,anchor_losses
from .rewards.base import FakeGroupScorer,SLOTS
from .rewards.api_group import score_groups
from .rewards.qwen_group import QwenGroupScorer,CachedGroupScorer
from .phase_progress import PhaseProgress
from .rewards.monitor import record_audit,audit_count,append_audit
from .distributed import gradient_mean,gradients_hash,tensor_hash,tree_hash
from .data.provenance import require_current_group

def setup():
    rank=int(os.environ.get("RANK",0));size=int(os.environ.get("WORLD_SIZE",1));local=int(os.environ.get("LOCAL_RANK",0))
    torch.cuda.set_device(local)
    control=None
    if size>1:
        dist.init_process_group("nccl",timeout=datetime.timedelta(hours=2))
        control=dist.new_group(backend="gloo",timeout=datetime.timedelta(hours=2))
    return rank,size,local,control

def broadcast(obj,rank,size,control):
    if size==1:return obj
    items=[obj if rank==0 else None];dist.broadcast_object_list(items,src=0,group=control);return items[0]

def sync(size,control):
    if size>1:dist.barrier(group=control)

def schedule(samples,config,world_size,batch):
    """No real-condition padding. Every inner visit includes each valid video once."""
    result=[];accum=config["nft"]["accumulation"]
    for inner in range(config["nft"]["inner_epochs"]):
        ids=list(range(len(samples)));random.Random(config["plan"]["seed"]+batch*99991+inner).shuffle(ids)
        draws=config["rollout"]["steps"] if config["nft"]["full_schedule"] else config["nft"]["time_draws_per_visit"]
        visits=[(i,inner,draw) for i in ids for draw in range(draws)]
        width=world_size*accum
        for offset in range(0,len(visits),width):
            window=visits[offset:offset+width];valid=len(window);window += [None]*(width-valid)
            result.append({"visits":window,"valid":valid})
    return result

def formal_gate(run,stage):
    gate_path=run/"reports/integration_gate.json"
    if not gate_path.exists() or not read_json(gate_path).get("passed"):raise RuntimeError("Formal launch requires a passing integration_gate.json")
    gate=read_json(gate_path)
    from .io import ROOT,sha256
    for path,expected in gate.get("source_code_hashes",{}).items():
        if not (ROOT/path).is_file() or sha256(ROOT/path)!=expected:raise RuntimeError("RL source changed after acceptance; rerun the integration gate")
    for path,expected in gate.get('runtime_artifact_hashes',{}).items():
        if not (ROOT/path).is_file() or sha256(ROOT/path)!=expected:raise RuntimeError('RL configuration/prompt/script changed after acceptance')
    for path,expected in gate.get('protected_model_files',{}).items():
        full=ROOT/path
        if not full.is_file():raise RuntimeError('Accepted source model file is missing: '+path)
        stat=full.stat()
        if stat.st_size!=expected['size'] or stat.st_mtime_ns!=expected['mtime_ns']:raise RuntimeError('Accepted source model file changed; rerun complete source hash verification: '+path)
    if stage=="local":
        from .formal import stage_c_control
        if stage_c_control(run) is not None:raise RuntimeError('User requested stop after Stage B; Stage C is disabled')
        q=reward_path(run)/"reports/quality_gate.json"
        if not q.exists() or not read_json(q).get("passed"):raise RuntimeError("Formal C requires an independently passing reward quality gate")
        manifest=read_json(reward_checkpoint(run)/'manifest.json')
        if manifest.get('smoke') or read_json(q).get('version')!=manifest['version']:raise RuntimeError('Formal C needs the version-matched, non-smoke reward model')
        from .io import QWEN
        if Path(manifest['base_model']).resolve()!=QWEN.resolve():raise RuntimeError('Formal C reward base identity differs')
        profile=run/"reports/joint_profile.json"
        sequential=run/'reports/C_timeslice_acceptance.json'
        measured=(profile.exists() and read_json(profile).get('selected_topology'))
        sequential_verified=(sequential.exists() and read_json(sequential).get('passed'))
        if not measured and not sequential_verified:raise RuntimeError("Formal C topology must be measured")

def train(args,stage="api"):
    from .reproducibility import configure
    backend=configure()
    if os.environ.get("RL_POSTTRAIN_DEBUG_STACKS")=="1":
        import faulthandler
        faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
    c=validate(load_config(args.config),args.run_id);run=run_path(args.run_id)
    if not args.smoke:formal_gate(run,stage)
    if not args.smoke and (c['reward']['protocol']!='mechanics-reward-v2-24x832-124images' or c['data'].get('split_policy')!='source-action-temporal-v2'):raise ValueError('Formal training requires the current v2 teacher and source leakage protocol; legacy configurations are development-only')
    if args.smoke and args.max_optimizer_steps is None:raise ValueError("Smoke needs an explicit optimizer-step cap")
    if args.scorer=="fake" and not args.smoke:raise ValueError("Fake scorer is software-only")
    if c.get("development_smoke") and not args.smoke:raise ValueError("Development config cannot launch formal training")
    if args.allow_legacy_rendering and not args.smoke:raise ValueError('Legacy rendering is restricted to explicitly selected software smoke tests')
    if args.max_conditions is not None and (stage!='api' or not 0<args.max_conditions<=c['plan']['api_conditions']):raise ValueError('Trial cap must be a positive cumulative A condition count within the plan')
    rank,size,local,control=setup();torch.set_num_threads(4);torch.manual_seed(c["plan"]["seed"]+rank)
    # Global condition batches and the one-pass ledger do not depend on GPU count.
    records=catalog(c,run);by_id={r["condition_id"]:r for r in records};ledger=Ledger(run/"plan",records,c["plan"]["seed"])
    adapter=WorldAdapter(c,f"cuda:{local}");opt=optimizer(adapter.dit,c["nft"]);anchor=AnchorScheduler()
    forward=DistributedDataParallel(adapter.forward,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False) if size>1 else adapter.forward
    bank=PolicyBank(adapter.dit)
    frozen_baseline=parameter_hash(adapter.dit,True) if args.max_conditions is not None else None
    state={"optimizer_step":0,"batch_id":0,"update_cursor":0,"batch_done":True,"stage":stage,"smoke":args.smoke}
    latest=run/"world"/stage/"checkpoints/latest.json"
    checkpoint=Path(args.resume) if args.resume else Path(read_json(latest)["checkpoint"]) if latest.exists() else None
    if stage=="local" and checkpoint is None:checkpoint=world_checkpoint(run,'api')
    if checkpoint:
        state=load(checkpoint,adapter.dit,opt,bank,c,rank)
        if stage=='api' and state.get('stage')=='local':raise ValueError('A cannot resume from a C checkpoint')
        state["stage"]=stage
        # Repair the narrow crash window after checkpoint publication and before ledger commit.
        if state.get("batch_done") and ledger.state()["inflight"] and state["batch_id"]==ledger.state()["inflight"]["batch"]:
            if rank==0:ledger.commit(checkpoint)
            sync(size,control)
    state['backend']=backend
    if rank==0:write_json(run/"reports"/(stage+"_trainable.json"),adapter.audit)
    local_scorer=None
    if stage=='local' and rank==0 and args.phase!='generate':
        local_scorer=CachedGroupScorer(c,args.reward_version) if args.phase=='update' else QwenGroupScorer(c,args.reward_version)
    if stage=='local':
        selected_version=None
        if rank==0:
            selected_version=(args.reward_version or read_json(reward_checkpoint(run)/'manifest.json')['version']) if args.phase=='generate' else local_scorer.version
        reward_version=broadcast(selected_version,rank,size,control)
        if state.get('reward_version',reward_version)!=reward_version:raise ValueError('Frozen C reward model version changed during resume')
        state['reward_version']=reward_version
    started_steps=state["optimizer_step"]
    def publish(done=False):
        if os.environ.get("RL_POSTTRAIN_DEBUG_STACKS")=="1":print("TRACE",rank,"publish begin",state["optimizer_step"],flush=True)
        state["batch_done"]=done
        random_states=[None]*size if rank==0 else None
        if size>1:dist.gather_object(rng_state(),random_states,dst=0,group=control)
        else:random_states=[rng_state()]
        path=run/"world"/stage/"checkpoints"/f"step-{state['optimizer_step']:08d}-batch-{state['batch_id']:06d}-cursor-{state['update_cursor']:06d}.pt"
        if rank==0:
            save(path,adapter.dit,opt,bank,state.copy(),c,random_states)
            write_json(latest,{"checkpoint":str(path),"optimizer_step":state["optimizer_step"]})
        sync(size,control);return path
    while True:
        reservation=None
        if rank==0:
            existing=ledger.state()["inflight"]
            budget=c["plan"]["api_conditions"]-ledger.state()["api_consumed"] if stage=="api" else len(ledger.order)-ledger.state()["cursor"]
            if args.max_conditions is not None:budget=min(budget,args.max_conditions-ledger.state()['api_consumed'])
            if args.smoke and not existing:budget=min(budget,c["rollout"]["conditions_per_batch"])
            if existing:reservation=existing
            elif budget>0:
                count=min(c["rollout"]["conditions_per_batch"],budget)
                if stage=='local' and not args.smoke and ledger.state()['local_consumed']==0:
                    # Three generation chunks (24+24+8) remain one fixed-old audit
                    # buffer. No NFT step or cursor commit precedes its quality gate.
                    count=min(c['reward_model']['audit_initial_conditions'],budget)
                reservation=ledger.reserve(count,stage,bank.version)
                if stage=='local' and not args.smoke and ledger.state()['local_consumed']==0:
                    ledger.phase('reserved',initial_audit_buffer=True,ordinary_conditions_per_batch=c['rollout']['conditions_per_batch'],rollout_batch_count=math.ceil(count/c['rollout']['conditions_per_batch']))
                    reservation=ledger.state()['inflight']
        reservation=broadcast(reservation,rank,size,control)
        if reservation is None:break
        batch=reservation["batch"];state["batch_id"]=batch
        if reservation["old_version"]!=bank.version:raise RuntimeError("Zero policy lag required; in-flight old version differs")
        groups=[]
        if rank==0:
            for j,condition_id in enumerate(reservation["condition_ids"]):groups.append(str(prepare_group(by_id[condition_id],c,run,bank.version,reservation.get("global_condition_offset",batch*c["rollout"]["conditions_per_batch"])+j)))
        groups=[Path(p) for p in broadcast(groups,rank,size,control)]
        own=groups[rank::size]
        initial=sum(sum(candidate['complete'] for candidate in read_json(g/'manifest.json')['candidates']) for g in own)
        generation=PhaseProgress(run,batch,'generate',rank,4*len(own),initial) if args.phase!='update' else None
        for group in own:
            if args.phase!='update':
                adapter.generate_group(group,bank,args.interrupt_after_candidates,allow_legacy=args.allow_legacy_rendering,
                    candidate_batch_size=c.get('stage_c',{}).get('generation_batch_size',1),progress=generation.advance)
                if not read_json(group/"manifest.json").get("frame_assets"):complete_protocol(group,c)
            elif not all(candidate['complete'] for candidate in read_json(group/'manifest.json')['candidates']):
                raise RuntimeError('Generation phase is incomplete')
            else:
                from .io import sha256
                for candidate in read_json(group/'manifest.json')['candidates']:
                    directory=group/'candidates'/candidate['video_id']
                    if sha256(directory/'clean_latents.safetensors')!=candidate['latent_sha256'] or sha256(directory/'video.mp4')!=candidate['video_sha256']:
                        raise RuntimeError('Update input changed after generation')
        if generation:generation.finish()
        if generation:
            write_json(run/'progress'/f'batch-{batch:06d}'/f'generate-memory-rank-{rank}.json',
                {'rank':rank,'gpu':local,'candidate_batch_size':c.get('stage_c',{}).get('generation_batch_size',1),
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        sync(size,control)
        if args.phase=='generate':
            if rank==0:ledger.phase('generated')
            publish(False)
            sync(size,control)
            if size>1:dist.destroy_process_group()
            return
        normalization_path=run/"plan"/f"batch-{batch:06d}-normalization.json"
        scores=None;score_kind=reservation["stage"]
        if rank==0:
            ledger.phase("generated")
            if normalization_path.exists():scores=read_json(normalization_path)["scores"]
            elif args.scorer=="fake":scores=[FakeGroupScorer().score(g) for g in groups]
            elif score_kind=="api":scores=score_groups(c,run,groups,args.api_group_limit)
            else:scores=[local_scorer.score(g) for g in groups]
            if not normalization_path.exists():
                points=[[s[k] for k in SLOTS] for s in scores];n=c["nft"]
                normalized=transform(points,adv_clip=n["adv_clip"],std_floor=n["std_floor"],normalization=n["normalization"],reward_mode=n["reward_mode"],guard_min_max=n["guard_min_max"],guard_min_spread=n["guard_min_spread"])
                write_json(normalization_path,{"scores":scores,"normalization":{k:v.tolist() for k,v in normalized.items()},"scorer":args.scorer,"old_version":bank.version,"stage":stage,"condition_ids":reservation['condition_ids']})
            ledger.phase("scored")
        scores=broadcast(scores,rank,size,control);sync(size,control)
        if args.interrupt_after_scoring:raise InterruptedError("Injected interruption after durable scoring, before any update")
        data=read_json(normalization_path);normalized=data["normalization"]
        # Teacher/RM audits consume this C reservation, never substitute new conditions.
        if stage=="local" and not args.smoke and score_kind=="local":
            monitor_path=run/"metrics/reward_audits.json"
            monitor_state=read_json(monitor_path) if monitor_path.exists() else {}
            requested_audits=audit_count(monitor_state,ledger.state(),c,len(groups))
            if reservation.get('initial_audit_buffer'):
                gate=None
                if rank==0:
                    teacher=score_groups(c,run,groups,args.api_group_limit)
                    from .rewards.initial_audit import assess
                    gate=assess(run,c,groups,teacher,scores,reward_version)
                    if gate['passed']:
                        append_audit(monitor_path,groups,[[s[k] for k in SLOTS] for s in teacher],[[s[k] for k in SLOTS] for s in scores],c['reward_model']['audit_initial_conditions'],c['reward_model']['audit_groups'])
                gate=broadcast(gate,rank,size,control)
                if not gate['passed']:
                    publish(False);raise RuntimeError('C initial 56-condition gate: '+gate['status']+'; saved buffer and checkpoint retained')
            elif requested_audits:
                monitor=None
                if rank==0:
                    audit_groups=groups[:requested_audits]
                    teacher=score_groups(c,run,audit_groups,args.api_group_limit)
                    monitor=append_audit(monitor_path,audit_groups,[[s[k] for k in SLOTS] for s in teacher],[[s[k] for k in SLOTS] for s in scores[:len(audit_groups)]],c["reward_model"]["audit_initial_conditions"],c["reward_model"]["audit_groups"])
                monitor=broadcast(monitor,rank,size,control)
                if monitor["paused"]:
                    publish(False);raise RuntimeError("Reward drift monitor paused C; checkpoint retained")
        samples=[]
        for i,g in enumerate(groups):
            require_current_group(g,c,args.allow_legacy_rendering)
            if normalized["keep"][i]:
                m=read_json(g/"manifest.json")
                for j,candidate in enumerate(m["candidates"]):samples.append({"group":str(g),"candidate":candidate["video_id"],"r":normalized["r"][i][j]})
        windows=schedule(samples,c,size,batch)
        cursor=state["update_cursor"] if not state["batch_done"] else 0
        if state.get("schedule_world_size",size)!=size and cursor:raise RuntimeError("Mid-batch topology change requires finishing the batch in its original topology")
        state["batch_done"]=False;state["schedule_world_size"]=size
        updating=PhaseProgress(run,batch,'update',rank,len(windows),cursor) if rank==0 else None
        for wi in range(cursor,len(windows)):
            window=windows[wi];valid=window["valid"];opt.zero_grad(set_to_none=True);metrics=[]
            replay={'optimizer_step':state['optimizer_step']+1,'batch':batch,'window':wi,'old_version':bank.version,'old_hash':tree_hash(bank.old),'ref_hash':tree_hash(bank.ref),'optimizer_before_hash':tree_hash(opt.state_dict()),'visits':[]} if args.audit_replay else None
            for mi in range(c["nft"]["accumulation"]):
                visit=window["visits"][mi*size+rank]
                sample=samples[visit[0]] if visit else {"group":str(groups[0]),"candidate":read_json(groups[0]/"manifest.json")["candidates"][0]["video_id"],"r":.5}
                g=Path(sample["group"]);values=adapter.condition_tensors(g)
                x0=load_file(g/"candidates"/sample["candidate"]/"clean_latents.safetensors")["clean"].to(adapter.device)
                seed=c["plan"]["seed"]+batch*1000003+wi*1009+rank*101+mi
                generator=torch.Generator("cpu").manual_seed(seed)
                idx=visit[2] if visit and c["nft"]["full_schedule"] else int(torch.randint(len(adapter.scheduler.sigmas),(1,),generator=generator))
                sigma=float(adapter.scheduler.sigmas[idx]);noise=torch.randn(x0.shape,generator=generator).to(adapter.device,x0.dtype)
                xt,_=forward_noise(x0,noise,sigma,values["first"])
                with bank.use("old"),torch.no_grad():old=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],False)["pred_video"].detach()
                ref=None
                if c["nft"]["ref_velocity_weight"]:
                    with bank.use("ref"),torch.no_grad():ref=adapter.forward(xt,sigma,values["context"],values["actions"],values["first"],False)["pred_video"].detach()
                ctx=forward.no_sync() if size>1 else nullcontext()
                with ctx,bank.live_graph():
                    live=forward(xt,sigma,values["context"],values["actions"],values["first"],True)["pred_video"]
                    loss,terms=nft_loss(live,old,xt,x0,sigma,[sample["r"]],beta=c["nft"]["beta"],adv_clip=c["nft"]["adv_clip"],mode=c["nft"]["loss"],noise=noise)
                    if ref is not None:loss=loss+c["nft"]["ref_velocity_weight"]*(live[:,:,1:].float()-ref[:,:,1:].float()).square().mean()
                    scaled=loss*(size/valid if visit else 0)
                    scaled.backward()
                if visit:metrics.append({"nft":float(loss.detach()),"positive":float(terms["positive"].mean()),"negative":float(terms["negative"].mean()),"r":sample["r"],"sigma":sigma})
                if replay is not None:replay['visits'].append({'micro':mi,'visit':visit,'condition_id':read_json(g/'condition.json')['condition_id'],'candidate':sample['candidate'],'latent_hash':tensor_hash(x0),'conditioning_hash':tree_hash(values),'noise_seed':seed,'noise_hash':tensor_hash(noise),'timestep_index':idx,'sigma':sigma,'r':sample['r']})
            # One independent genuine train anchor per rank and optimizer update.
            anchor_seed=c["plan"]["seed"]+state["optimizer_step"]*77777+rank*31337
            anchor_rng=torch.Generator("cpu").manual_seed(anchor_seed)
            ar=records[int(torch.randint(len(records),(1,),generator=anchor_rng))]
            ag=prepare_group(ar,c,run/"conditions/anchors",bank.version,anchor_seed)
            av=adapter.condition_tensors(ag);sigma,weight=anchor.draw(anchor_rng)
            noise=torch.randn(av["real"].shape,generator=anchor_rng).to(adapter.device,av["real"].dtype)
            with (forward.no_sync() if size>1 else nullcontext()),bank.live_graph():
                fm,ce,_=anchor_losses(adapter,forward,av["real"],av["first"],av["context"],av["actions"],sigma,weight,noise)
                (c["nft"]["fm_anchor_weight"]*fm+c["nft"]["transition_anchor_weight"]*ce).backward()
            gradient_mean(adapter.dit)
            norms=gradient_norms(adapter.dit)
            grad=torch.nn.utils.clip_grad_norm_([p for p in adapter.dit.parameters() if p.requires_grad],c["nft"]["max_grad_norm"])
            if not torch.isfinite(grad):raise FloatingPointError("Nonfinite world gradient")
            if replay is not None:
                replay['anchor']={'condition_id':ar['condition_id'],'seed':anchor_seed,'sigma':float(sigma),'weight':float(weight),'noise_hash':tensor_hash(noise),'conditioning_hash':tree_hash(av)}
                replay['gradient_hash']=gradients_hash(adapter.dit)
                append_jsonl(run/f'metrics/replay_rank_{rank}.jsonl',replay)
            opt.step();opt.zero_grad(set_to_none=True);state["optimizer_step"]+=1;state["update_cursor"]=wi+1
            if rank==0:
                ledger.phase("updating",update_cursor=wi+1)
                append_jsonl(run/"metrics"/(stage+"_steps.jsonl"),{"optimizer_step":state["optimizer_step"],"batch":batch,"update_cursor":wi+1,"old_version":bank.version,"normalization":normalized,"rank0_grad_norm":norms,"rank0_fm":float(fm.detach()),"rank0_ce":float(ce.detach()),"rank0_samples":metrics,"global_valid_videos":valid,"world_size":size,"mode":c["nft"]["mode"],"peak_allocated":torch.cuda.max_memory_allocated(),"smoke":args.smoke})
                print(stage,"optimizer",state["optimizer_step"],"global valid",valid,"grad",norms,flush=True)
            publish(False)
            if updating:updating.advance()
            if args.interrupt_after_optimizer_step==state["optimizer_step"]:raise InterruptedError("Injected stop at committed optimizer boundary")
            if args.max_optimizer_steps is not None and state["optimizer_step"]-started_steps>=args.max_optimizer_steps and wi+1<len(windows):
                if rank==0:print("Optimizer cap reached mid-batch; resume keeps old policy and exact cursor",flush=True)
                sync(size,control)
                if size>1:dist.destroy_process_group()
                return
        # An entirely filtered batch consumes its conditions, but performs no
        # optimizer step and must not move the sampling policy a second time.
        eta=bank.update_old(state["optimizer_step"]) if windows else state.get('ema_eta',0.)
        state["update_cursor"]=0;state["ema_eta"]=eta;state['all_filtered_batch']=not bool(windows)
        if args.audit_replay:append_jsonl(run/f'metrics/replay_rank_{rank}.jsonl',{'event':'ema','optimizer_step':state['optimizer_step'],'eta':eta,'old_version':bank.version,'old_hash':tree_hash(bank.old),'ref_hash':tree_hash(bank.ref)})
        committed=publish(True)
        if rank==0:ledger.commit(committed)
        if updating:updating.finish()
        sync(size,control)
        if args.smoke or args.single_batch or (args.max_optimizer_steps is not None and state["optimizer_step"]-started_steps>=args.max_optimizer_steps):break
    hashes=[None]*size if size>1 else [parameter_hash(adapter.dit,False)]
    if size>1:dist.all_gather_object(hashes,parameter_hash(adapter.dit,False),group=control)
    if len(set(hashes))!=1:raise AssertionError("World trainable parameters differ across DDP ranks")
    if rank==0:write_json(run/"reports"/(stage+"_ddp_sync.json"),{"ranks":size,"trainable_hashes":hashes,"equal":True,"mode":c["nft"]["mode"]})
    final_random=[None]*size if rank==0 else None
    if size>1:dist.gather_object(rng_state(),final_random,dst=0,group=control)
    else:final_random=[rng_state()]
    if rank==0:
        final=run/"world"/stage/"final/versions"/f"conditions-{ledger.state()['cursor']:08d}-step-{state['optimizer_step']:08d}"/"world.pt"
        if not final.exists():save(final,adapter.dit,opt,bank,state,c,final_random)
        write_json(run/'world'/stage/'final/current.json',{'checkpoint':str(final),'optimizer_step':state['optimizer_step'],'conditions_cursor':ledger.state()['cursor']})
        canonical=run/"world"/stage/"final/world.pt"
        if not canonical.exists() or canonical.is_symlink():
            publish_link(canonical,final);publish_link(canonical.with_suffix(".json"),final.with_suffix(".json"))
        completion={"software_smoke":args.smoke,"formal_quality":False,"optimizer_step":state["optimizer_step"],"checkpoint":str(final),"ledger":ledger.state(),"old_version":bank.version,'planned_A_complete':ledger.state()['api_consumed']>=c['plan']['api_conditions'],'trial_cap':args.max_conditions}
        if frozen_baseline is not None:
            after=parameter_hash(adapter.dit,True);assert after==frozen_baseline,'Frozen parameters changed during A trial'
            completion['frozen_hash']=after;completion['parameter_delta_from_ref_l2']=sum(float((p.detach().cpu()-bank.ref[n]).square().sum()) for n,p in bank.params.items())**.5
        write_json(run/"reports"/(stage+"_completion.json"),completion)
        print(f"{stage} finished: {final}",flush=True)
    sync(size,control)
    if size>1:dist.destroy_process_group()

def parser():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--run-id",required=True);p.add_argument("--resume");p.add_argument("--smoke",action="store_true");p.add_argument("--max-optimizer-steps",type=int);p.add_argument("--scorer",choices=["fake","api","local"],default="api");p.add_argument("--api-group-limit",type=int);p.add_argument("--reward-version");p.add_argument("--interrupt-after-candidates",type=int);p.add_argument("--interrupt-after-scoring",action="store_true");p.add_argument("--interrupt-after-optimizer-step",type=int);p.add_argument('--audit-replay',action='store_true');p.add_argument('--allow-legacy-rendering',action='store_true');p.add_argument('--max-conditions',type=int)
    p.add_argument('--phase',choices=['all','generate','update'],default='all');p.add_argument('--single-batch',action='store_true')
    return p
