from graph import graph
import json

def parse_json(path_to_file):
    """Return subclone environment as json dictionary"""
    with open(path_to_file) as f:
        data = json.load(f)
    
    return data

if __name__ == "__main__":
    
    env = parse_json('graph/sub_env.json')
    
    g = graph.Graph(env)
    print(f'Graph : {g.tag}')