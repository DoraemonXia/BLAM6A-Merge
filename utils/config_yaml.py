import yaml
def load_yaml(file):
    with open(file,'r',encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    return conf

def update_yaml(file,value,key1,key2=None):
    with open(file, encoding="utf-8") as f:
        data = yaml.load(f, yaml.FullLoader)
        if key2==None:
            data[key1] = value
        else:
            data[key1][key2] = value
    with open(file, "w", encoding="utf-8") as f:
        yaml.dump(data, f)