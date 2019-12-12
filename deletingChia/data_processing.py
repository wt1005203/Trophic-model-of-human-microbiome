########### Self-customized setting
import pandas as pd
import numpy as np

########### Load Chia network as net (containing information of metabolite consumption and production)
net = pd.read_csv('pruned_chia_network.csv')
mean_net = net.groupby('microbes_ID').mean()
selfish_net = mean_net[mean_net.iloc[:,1] == 2]
i_selfish = selfish_net.index.values   #### i_selfish returns IDs of microbes don't generate byproducts

########### Load names of all nodes in the Chia network
names = pd.read_csv('names_ID.txt',sep=': ')
names.set_index('IDs', inplace=True)

########### Load names of all nodes in the Chia network
i_intake = pd.read_csv('nutrient_intake_ID.txt',sep=': ')
i_intake = i_intake['IDs'].values

########### Load all gut metagenomic data of all 41 individuals
thai_metagenome_all = pd.read_csv('abundance_matched_thai.txt', sep='\t')
thai_metagenome_all.head()
thai_metagenome_all = thai_metagenome_all.groupby('Chia_id').sum().iloc[1:,].reset_index()
thai_metagenome_ID = thai_metagenome_all['Chia_id']
#print((thai_metagenome_ID!=0).sum())
thai_metagenome = thai_metagenome_all[thai_metagenome_ID!=0].iloc[:,3:]
thai_metagenome_ID = thai_metagenome_ID[thai_metagenome_ID!=0]

########### Load all gut metabolome data of all 41 individuals
thai_metabolome_all = pd.read_excel('metabolome_matched_thai_modified_by_Tong.xlsx')
thai_metabolome_all = thai_metabolome_all.groupby('Chia_id').sum().iloc[1:,].reset_index()
thai_metabolome_ID = thai_metabolome_all['Chia_id']
#print((thai_metabolome_ID!=0).sum())
thai_metabolome = thai_metabolome_all[thai_metabolome_ID!=0].iloc[:,2:]
thai_metabolome_ID = thai_metabolome_ID[thai_metabolome_ID!=0]

intersected_names = np.intersect1d(thai_metagenome.columns.values, thai_metabolome.columns.values)
thai_metagenome = thai_metagenome[intersected_names]
thai_metabolome = thai_metabolome[intersected_names]

i_nonzero_microbes = thai_metagenome_ID.values.copy()
i_nonzero_metabolites = net['metabolites_ID'].unique()
i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

########### pickle all processed data which are useful for simulations
import pickle

net_chia = net.copy()
pDel_list = np.arange(0.0, 1.0, 0.1)
for roundpDel in range(len(pDel_list)):
    pDel = pDel_list[roundpDel]

    net = net_chia.copy()
    i_x = np.random.choice(range(len(net_chia)), int(len(net_chia) * pDel), replace=False)
    net.drop(net.index[i_x], inplace=True)

    pickle_out = open("Chia_network"+str(roundpDel)+".pickle","wb")
    pickle.dump([net, i_selfish, i_intake, names], pickle_out)
    pickle_out.close()

    pickle_out = open("Thai_data"+str(roundpDel)+".pickle","wb")
    pickle.dump([thai_metagenome_ID, thai_metagenome, thai_metabolome_ID, thai_metabolome, i_nonzero_metabolites], pickle_out)
    pickle_out.close()