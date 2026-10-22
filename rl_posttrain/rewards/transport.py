"""Optional SSE transport preserving the canonical teacher request and raw events."""
import json
import time
from ..io import write_json, append_jsonl, now, teacher_path


class SubmissionIncomplete(RuntimeError):
    """The upstream may have processed/billed an interrupted streamed request."""


def submit(client, url, headers, request, group, attempt, streaming=False, max_elapsed=900):
    def decoded(response):
        try:
            return response.json(), True
        except ValueError:
            return None, False
    if not streaming:
        response = client.post(url, headers=headers, json=request)
        raw, valid = decoded(response)
        return response, raw, valid, {'response_transport': 'json'}
    wire = dict(request, stream=True, stream_options={'include_usage': True})
    teacher=teacher_path(group)
    write_json(teacher/'transport_requests'/f'attempt_{attempt:02d}.json', wire)
    events = teacher/f'stream_attempt_{attempt:02d}.jsonl'
    started = time.monotonic()
    with client.stream('POST', url, headers=headers, json=wire) as response:
        write_json(teacher/f'stream_headers_attempt_{attempt:02d}.json', {
            'timestamp': now(), 'http_status': response.status_code,
            'content_type': response.headers.get('content-type'),
            'provider_request_id': response.headers.get('x-request-id'),
            'cf_ray': response.headers.get('cf-ray'),
            'headers_latency_seconds': time.monotonic()-started})
        if response.status_code >= 400 or 'text/event-stream' not in response.headers.get('content-type', ''):
            response.read()
            raw, valid = decoded(response)
            return response, raw, valid, {'response_transport': 'json_stream_requested'}
        text = []
        completed = False
        finish = None
        usage = None
        metadata = {}
        for line in response.iter_lines():
            # Retain comments/heartbeats and complete provider chunks, even if
            # the process later dies before a complete four-score JSON exists.
            append_jsonl(events, {'timestamp': now(), 'line': line})
            if time.monotonic()-started>max_elapsed:
                raise SubmissionIncomplete('Stream exceeded its elapsed time limit')
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                completed = True
                break
            try:
                chunk = json.loads(data)
                if chunk.get('error'):
                    raise SubmissionIncomplete('Provider reported a stream error')
                for key in ('id', 'created', 'model', 'system_fingerprint'):
                    if key in chunk:
                        metadata[key] = chunk[key]
                if chunk.get('usage') is not None:
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise SubmissionIncomplete('Unexpected multiple completion choices')
                    content = choice.get('delta', {}).get('content')
                    if content is not None:
                        if not isinstance(content, str):
                            raise SubmissionIncomplete('Unexpected non-text completion delta')
                        text.append(content)
                    if choice.get('finish_reason') is not None:
                        finish = choice['finish_reason']
            except (ValueError, TypeError, AttributeError) as error:
                raise SubmissionIncomplete('Malformed provider stream chunk') from error
        if not completed or finish != 'stop':
            raise SubmissionIncomplete('Stream did not complete with stop and DONE')
        raw = dict(metadata, object='chat.completion', usage=usage,
                   choices=[{'index': 0, 'message': {'role': 'assistant', 'content': ''.join(text)}, 'finish_reason': finish}])
        return response, raw, True, {'response_transport': 'chat_sse_v1',
                                    'assembled_from_stream': True, 'raw_stream_file': str(events)}
