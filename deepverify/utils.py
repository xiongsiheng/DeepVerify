import json
from rich import print as rprint
from json_repair import repair_json

def json_loads_robust(json_str):
    try:
        return json.loads(json_str)
    except Exception as e:
        pass
    
    data = json_str.split('```')[-2]
    if data.startswith('json'):
        data = data[len('json'):]
    
    try:
        return json.loads(data)
    except Exception as e:
        try:
            rprint(f"[yellow]Error parsing JSON ... attempting repair [/yellow]")
            return json.loads(repair_json(data))
        except Exception as e:
            rprint(f"[red]Error parsing JSON ... attempting repair [/red]")
            print(data)
            raise e
