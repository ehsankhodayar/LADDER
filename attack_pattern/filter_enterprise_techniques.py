import os.path

import fire
import pandas as pd


def filter_enterprise_techniques(default_dataset, destination_dir, filter_techniques):
    df_default = pd.read_csv(default_dataset, keep_default_na=False)
    filtered_ttps = list(pd.read_csv(filter_techniques, keep_default_na=False).iloc[:, 0])

    previous_id = ''

    technique_ids = []
    technique_names = []
    technique_descriptions = []

    for row_default in df_default.itertuples(index=False):
        technique_id = row_default[0]

        if technique_id == '':
            sub_technique_id = row_default[1][1:]
            technique_id = previous_id + sub_technique_id
        else:
            previous_id = technique_id

        name = row_default.__getattribute__('Name')
        description = row_default.__getattribute__('Description')

        if technique_id in filtered_ttps:
            technique_ids.append(technique_id)
            technique_names.append(name)
            technique_descriptions.append(description)

    uncovered_ttps = []
    for filtered_ttp in filtered_ttps:
        if filtered_ttp not in technique_ids:
            uncovered_ttps.append(filtered_ttp)

    if uncovered_ttps:
        print('The following filtered technique IDs were not found in the dataset:')
        for uncovered_ttp in uncovered_ttps:
            print('\tCompleting ', uncovered_ttp)
            name = input('Technique Name (Remove line breaks and links): ')
            description = input('Technique Description (Remove line breaks and links): ')

            if name and description:
                technique_ids.append(uncovered_ttp)
                technique_names.append(name)
                technique_descriptions.append(description)

                print(f'Technique ID: {uncovered_ttp} updated successfully!')
            else:
                raise Exception("Invalid Technique Name or Description!")

        # Save the new enterprise technique dataset
        df_new = pd.DataFrame({
            'ID': technique_ids,
            'Name': technique_names,
            'Description': technique_descriptions
        })

        print(f"Saving the new enterprise technique dataset in {destination_dir}...")
        df_new.to_csv(os.path.join(destination_dir, 'enterprise_techniques_customized.csv'), index=False)

fire.Fire(filter_enterprise_techniques)
