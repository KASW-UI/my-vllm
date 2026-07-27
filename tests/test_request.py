from my_vllm.engine.request import Request, RequestStatus


def test_request_lifecycle():
    request = Request(request_id="lifecycle-test")
    request.prompt_token_ids = [2, 100, 200, 300]

    assert request.status == RequestStatus.WAITING
    assert request.request_id == "lifecycle-test"
    assert len(request.prompt_token_ids) == 4
    assert len(request.output_token_ids) == 0

    # Simulate RUNNING
    request.status = RequestStatus.RUNNING
    request.append_output_token_ids(400)
    request.append_output_token_ids(500)
    assert request.status == RequestStatus.RUNNING
    assert len(request.output_token_ids) == 2

    # Simulate FINISHED
    request.status = RequestStatus.FINISHED_STOPPED
    assert request.status == RequestStatus.FINISHED_STOPPED

    print(f"Request: {request}")
