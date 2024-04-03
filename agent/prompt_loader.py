import json
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.load.load import loads
from langchain_core.prompts import BasePromptTemplate


def read_dict_from_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def load_prompt_from_file (file_path) -> Any:
        res_dict = read_dict_from_file(file_path)      
        obj = loads(json.dumps(res_dict["manifest"]))        
        if isinstance(obj, BasePromptTemplate):
            if obj.metadata is None:
                obj.metadata = {}
            obj.metadata["lc_hub_owner"] = res_dict["owner"]
            obj.metadata["lc_hub_repo"] = res_dict["repo"]
            obj.metadata["lc_hub_commit_hash"] = res_dict["commit_hash"]
        return obj
    
   