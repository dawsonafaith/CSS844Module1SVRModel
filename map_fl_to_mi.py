# format the data to combine GDD information, veg indices, Accession data, some of the FL flowering data, and stand count
import pandas as pd

mi_data = pd.read_csv('MI23_sweetcap_metadata.csv')
fl_data1 = pd.read_csv('sweetCAP_2022_flowering_related_traits.csv')
mi_data_stand_count = pd.read_csv('Michigan_2023_StandCountAndTarSpot(Sheet1).csv')
#fl_data2 = pd.read_csv('sweetCAP_2022_pheno_raw.csv')

veg_indices = pd.read_csv('vegetation_indices_results_2.csv')

mi_vars = set(mi_data['Accession'])
fl_vars1 = set(fl_data1['Maternal_Genotype'])
#fl_vars2 = set(fl_data2['Name '])

overlap_vars = list(mi_vars.intersection(fl_vars1))

# check how much overlap there is in mi & fl varities - i think there may be duplicates in the fl vars ?
print(len(mi_vars.intersection(fl_vars1)))

#create dictionary that maps MI plot:Accession
mi_plot_to_accession = mi_data.set_index('Plot')['Accession'].to_dict()

# add the column 'Accession' to veg indices dataframe - uses dictionary defined above to map accession to plot id
veg_indices['Accession'] = veg_indices.apply(lambda row: mi_plot_to_accession[row.id], axis=1)

# create dictionary that maps MI plot:stand count - then add column to veg indices
mi_plot_to_stand_count = mi_data_stand_count.set_index('plot')['StandCount1'].to_dict()
veg_indices['Stand Count'] = veg_indices.apply(lambda row: mi_plot_to_stand_count[row.id], axis=1)

# create dictionaries that map maternal genotype(accession):flowering trait
fl_accession_to_50_percent_pollen = fl_data1.set_index('Maternal_Genotype')['GGD to 50% Pollen'].to_dict()
fl_accession_to_50_percent_silk = fl_data1.set_index('Maternal_Genotype')['GDD to 50% Silk'].to_dict()

# add columns 'FL GDD to 50% Pollen' and 'FL GDD to 50% Silk' to VI df 
# lambda checks that the accession is present in both MI and FL first - if not present in both then 'None' inserted (these lines removed from df below)
veg_indices['FL GDD to 50% Pollen'] = veg_indices.apply(lambda row: fl_accession_to_50_percent_pollen[row.Accession] if row.Accession in overlap_vars else None , axis=1)
veg_indices['FL GDD to 50% Silk'] = veg_indices.apply(lambda row: fl_accession_to_50_percent_silk[row.Accession] if row.Accession in overlap_vars else None , axis=1)

# remove and NaNs (if there wasn't flowering data for a given accession)
#print(len(veg_indices))
veg_indices = veg_indices.dropna()
#print(len(veg_indices))

# write to csv
veg_indices.to_csv('MI23 and FL22 VI Accession Flowering StandCount.csv', index=False)