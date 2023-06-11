import yaml
import argparse
# with open('data.yaml', encoding='UTF-8') as yaml_file:
#   data = yaml.safe_load(yaml_file)
# print(type(data))
# print(data)
def load_cfg_from_yaml_str(str_obj):
    """Load a config from a YAML string encoding."""
    with open(str_obj, encoding='UTF-8') as f:
        cfg_as_dict = yaml.safe_load(f)
    return cfg_as_dict

cfg = load_cfg_from_yaml_str('/lustre/home/acct-seesb/seesb-user1/hs/superResolution/mscmr/componets/fastsurfer/config1.yaml')
paraser =argparse.ArgumentParser(description='McMRSR')
paraser.set_defaults(**cfg)

arg = paraser.parse_args()
print(arg)
