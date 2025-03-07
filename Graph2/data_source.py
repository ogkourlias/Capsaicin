
import yaml
def get_data():
    with open('data_path.yaml','r') as data_file:
        return yaml.safe_load(data_file)