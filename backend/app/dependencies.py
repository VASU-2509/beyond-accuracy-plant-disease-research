from fastapi import Request

def settings(request: Request):
    return request.app.state.settings

def inference(request: Request):
    return request.app.state.inference
