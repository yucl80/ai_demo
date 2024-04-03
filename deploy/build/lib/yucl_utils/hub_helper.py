
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.prompts import BasePromptTemplate
import os

if TYPE_CHECKING:
    from langchainhub import Client

def _get_client(api_url: Optional[str] = None, api_key: Optional[str] = None) -> Client:
    try:
        from langchainhub import Client
    except ImportError as e:
        raise ImportError(
            "Could not import langchainhub, please install with `pip install "
            "langchainhub`."
        ) from e

    # Client logic will also attempt to load URL/key from environment variables
    return Client(api_url, api_key=api_key) 

_prompt_hub_cache_path_ = "/home/test/src/code/prompts/" 

def _write_dict_to_file(res_dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(res_dict, file, ensure_ascii=False, indent=4)

def _read_dict_from_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def _download_prompt(
    owner_repo_commit: str,
    file_base_path: str,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:   
    client = _get_client(api_url=api_url, api_key=api_key)

    if hasattr(client, "pull_repo"):
        # >= 0.1.15
        res_dict = client.pull_repo(owner_repo_commit)
        file_path = file_base_path + owner_repo_commit.split("/")[0]       
        if not os.path.exists(file_path):
            os.makedirs(file_path)      
        _write_dict_to_file(res_dict, file_base_path + owner_repo_commit + ".json")

def _load_prompt_from_file (file_path) -> Any:
        res_dict = _read_dict_from_file(file_path)      
        obj = loads(json.dumps(res_dict["manifest"]))        
        if isinstance(obj, BasePromptTemplate):
            if obj.metadata is None:
                obj.metadata = {}
            obj.metadata["lc_hub_owner"] = res_dict["owner"]
            obj.metadata["lc_hub_repo"] = res_dict["repo"]
            obj.metadata["lc_hub_commit_hash"] = res_dict["commit_hash"]
        return obj      

def pull_repo(owner_repo_commit: str) -> Any:
    if os.path.exists(_prompt_hub_cache_path_ + owner_repo_commit + ".json"):
        prompt = _load_prompt_from_file(_prompt_hub_cache_path_ + owner_repo_commit + ".json")
        return prompt
    else:
        _download_prompt(owner_repo_commit, _prompt_hub_cache_path_)
        prompt = _load_prompt_from_file(_prompt_hub_cache_path_ + owner_repo_commit + ".json")
        return prompt
   
#prompt = download_prompt("hwchase17/react-chat-json")

