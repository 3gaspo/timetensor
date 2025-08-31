from datasets import get_dataset_config_names, load_dataset

#configs = get_dataset_config_names("Salesforce/lotsa_data")
#print(configs)

ds = load_dataset("Salesforce/lotsa_data", 'bdg-2_bear')

print(type(ds))
print(len(ds))
print(ds.shape)
