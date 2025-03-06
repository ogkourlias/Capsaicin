
import yaml
def get_data():
    with open('scripts/data_spicy.yaml','r') as data_file:
        return yaml.safe_load(data_file)