from huggingface_hub import snapshot_download
import os

def DownloadModel():
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

    snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="../Qwen2.5-7B-Instruct"
    )
if __name__== "__main__":
    DownloadModel()