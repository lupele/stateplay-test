# Stage C：三阶段轮流使用全部所选 GPU

生成视频 → 冻结 Qwen 奖励评分 → NFT 参数更新依次执行。每一阶段都在
`--gpus` 指定的全部卡上启动进程；评分是每卡一个完整多模态模型副本，
不同卡处理不同条件组。前一阶段的进程退出、释放显存后才启动下一阶段。
评分期间不保留世界模型，更新期间不保留 Qwen，不再预留奖励模型专卡。
小于 GPU 数的尾批只需部分卡处理真实条件，不能补抽样本凑数。

这种方式增加每轮加载模型的时间。总计时包括加载、进程切换和 checkpoint 保存，
不承诺一定快于常驻模型方案；实际收益看完整批次的端到端吞吐。

## 启动参数

```bash
outputs/rl_posttrain/_envs/world/bin/python -m rl_posttrain.prepare_local_config \
  --run-id study_v1 --gpus 0,1,2,3 \
  --generation-batch-size 2 --reward-batch-size 2

# 将上一步打印的配置路径填入 --config。
bash rl_posttrain/scripts/run_local.sh \
  --run-id study_v1 \
  --config outputs/rl_posttrain/study_v1/plan/formal_C_timeslice_<hash>.yaml \
  --gpus 0,1,2,3 \
  --generation-batch-size 2 \
  --reward-batch-size 2
```

这是未来正式运行示例，不表示已启动。仍检查原有软件验收、奖励模型人工质量门槛
以及用户要求的 C 暂停，不会绕过尚未完成的人工审核。
`prepare_local_config` 从 A 最终 checkpoint 的配置建立配置，保持
world/nft/reward/data 及采样协议一致，默认设 `rollout.conditions_per_batch: 28`，
即每批 28 个条件、112 个候选视频；可用 `--conditions-per-batch` 显式覆盖。

- `--gpus`：实际 GPU 编号，支持非连续列表，如 `1,3,4,6`。四张 A100 一般为
  `0,1,2,3`；逻辑 rank 按该列表顺序映射。未指定时优先使用
  `CUDA_VISIBLE_DEVICES`，否则取配置中 world/reward 卡号的并集。
- `--generation-batch-size`：每卡同一个条件同时生成几个候选，范围 1～4。
  每组仍严格为四个不同 seed；例如 2 表示生成两批，绝非把每组视频数改为两个。
  VAE 解码仍逐候选执行。默认 1。
- `--reward-batch-size`：每卡一次处理几个完整评分请求。每请求保持 124 张图、
  非 thinking 提示词及四分数输出。批量请求分别左填充，输出逐行解析；不会把
  两组拼成一个上下文。默认 1。
- `--dry-run`：只打印卡号及 batch/预算设置，不启动任何模型或 API。

训练 microbatch 仍为每卡 1 个候选，NFT 梯度累积仍为 4，inner epochs 仍为 2。
物理 GPU 数会影响一次优化器更新的全局样本量，但不会改变条件预算。
条件批次数不再随卡数强制设为 `4×卡数`，默认按 28 组运行；四卡时每卡
生成七个条件组。首次 56 个条件的独立审计缓冲仍保留。

## 一次遍历与恢复

正式 study_v1 始终沿用原 epoch_order 和 cursor：总计 10,000 个条件，A 已消费
2,280，C 剩余 7,720。切换卡数、推理 batch 或测试不会重置正式 cursor，B 不消费
新条件。最后不足 28 组按实际剩余数处理。

每个生成阶段退出前保存 live/old/ref、optimizer、每 rank RNG 及批次状态；
评分输出按模型版本和完整输入身份落盘。更新阶段从该状态继续，完成全部更新和
EMA 后才提交条件账本。中断重启时：已生成的不重生成，已评分的复用缓存，
部分更新继续原 update_cursor，不重复已提交更新；中途改变 world_size 仍被拒绝。
同一批中改变推理 batch 只影响未完成的任务，不改变已保存候选或评分。

## 进度与计时

终端同时显示四条进度：

1. A+C overall：已提交条件数/完整训练集条件数；ETA 在首个完整 C 批次后估计。
2. generate：当前批次已保存视频数/视频总数，显示 videos/s。
3. score：当前批次已完成完整评分请求数/条件数，显示 groups/s。
4. update：当前批次已提交优化器更新数/实际计划更新数，显示 steps/s。

各阶段 wall time 包含模型加载，work rate 从工作进度开始统计；缓存命中计入
完成数量，不伪装成新计算吞吐。总体 ETA 包括阶段切换和保存成本，但不能预测
未来 API 审计异常或等待人工审核的时间。首批完成前不凭空给出预计时间。

持久化文件：

- `progress/overall.json`：当前阶段、总进度、吞吐和预计完成时间。
- `progress/batch-XXXXXX/*-rank-*.json`：各 rank 的阶段进度。
- `progress/phases.jsonl`：各阶段完整耗时、GPU 列表、退出码及日志路径。
- `progress/batches.jsonl`：已完成批次的耗时及条件计数。
- `logs/C_batch-XXXXXX_{generate,score,update}.log`：真实 worker 日志。
- `plan/C_execution.json`：GPU、两个推理 batch、完整预算及模型版本。

后台运行时，可在另一个终端只读查看四个进度条：

```bash
outputs/rl_posttrain/_envs/world/bin/python -m rl_posttrain.watch_progress --run-id study_v1
```

完整请求 batch>1 时，保留原 SDPA 的因果注意力，按请求去除人工左 padding 后计算，
避免为两段约 4.4 万 token 的上下文创建巨大的稠密 padding 掩码。不裁剪真实 token、
图片或帧；线性注意力和投影/FFN 保持请求 batch。batch=1 继续使用原路径。
同样的 seed 在不同 batch 下可能产生浮点数差异，不能宣称视频逐位相同。

## 隔离验证

`verify_timeslice.py` 创建单独 `stagec_test_*` 目录，仅取 A 已消费的条件做测试，
使用 A 最终 live/old/ref、optimizer 和 RNG 的独立副本及正式 B 的只读模型。
低成本测试缩小画面、帧数和去噪步数，并关闭质量过滤以确保执行真实梯度；
不调用收费 API，不改正式 study_v1 账本。它只证明软件与恢复链路，不能作为
正式视频质量或 A100 性能验收。本机实际测试显卡是 L20。
