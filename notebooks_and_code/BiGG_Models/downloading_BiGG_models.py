import pandas as pd
import requests
import numpy as np
from os.path import join
import os
import warnings
warnings.filterwarnings("ignore")
from directory_infomation import *
from urllib.request import urlopen, Request

headers= {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
                      'AppleWebKit/537.11 (KHTML, like Gecko) '
                      'Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}


r = requests.get('http://bigg.ucsd.edu/api/v2/universal/metabolites')
bigg_metabolites = r.json() 
bigg_metabolites = bigg_metabolites["results"]
bigg_metabolites_list = [met["bigg_id"] for met in bigg_metabolites]

drugs_df = pd.read_pickle(join(datasets_dir, "substrate_synonyms", "KEGG_drugs_df.pkl"))
compounds_df = pd.read_pickle(join(datasets_dir, "substrate_synonyms", "KEGG_substrate_df.pkl"))
KEGG_substrate_df = compounds_df.append(drugs_df).reset_index(drop = True)
KEGG_substrate_df["substrate_lower"] = [sub.lower() for sub in KEGG_substrate_df["substrate"]]

def is_enzymatic_reaction(bigg_id, reaction_name):
    if bigg_id[-4:] in ["t2pp", "t3pp"]:
        return(False)
    elif bigg_id[:3] in ["EX_", "UP_", "DM_"]:
        return(False)
    elif (("exchange" in reaction_name) or ("transport" in reaction_name) or ("uptake" in reaction_name) 
          or ("biomass" in reaction_name.lower())):
        return(False)
    else:
        return(True)
    
def remove_non_enzymatic_reactions(df_reactions):
    droplist = []
    for ind in df_reactions.index:
        if not is_enzymatic_reaction(bigg_id = df_reactions["BiGG reaction ID"][ind], 
                                     reaction_name = df_reactions["reaction name"][ind]):
            droplist.append(ind)
    print("Removing %s non-enzymatic reactions" % len(droplist))
    df_reactions.drop(droplist, inplace = True)
    return(df_reactions)

def get_gene_reaction_rule(reaction):
    return(reaction["gene_reaction_rule"])

def get_direction(reaction):
    if reaction["lower_bound"] < 0 and reaction["upper_bound"] > 0:
        return("reversible")
    elif reaction["upper_bound"] > 0:
        return("forward")
    elif reaction["lower_bound"] < 0:
        return("backward")
    
def get_metabolites(reaction):
    return(reaction['metabolites'])

def get_EC(reaction):
    try:
        ec = reaction["annotation"]['ec-code']
        return(ec)
    except:
        return(np.nan)

def add_reaction_information(df_reactions, model_reactions, model_reactions_list):
    df_reactions["gene_reaction_rule"] = np.nan
    df_reactions["direction"] = np.nan
    df_reactions["metabolites"] = ""
    df_reactions["EC"] = ""
    
    for ind in df_reactions.index:
        BiGG_ID = df_reactions["BiGG reaction ID"][ind]
        reaction = model_reactions[model_reactions_list.index(BiGG_ID)]
        df_reactions["gene_reaction_rule"][ind] = get_gene_reaction_rule(reaction)
        df_reactions["direction"][ind] = get_direction(reaction)
        df_reactions["metabolites"][ind] = get_metabolites(reaction) 
        df_reactions["EC"][ind] = get_EC(reaction) 
    return(df_reactions)

def get_substrates_and_products(df_reactions):
    df_reactions["substrates"] = ""
    df_reactions["products"] = ""
    
    for ind in df_reactions.index:
        substrates = []
        products = []
        
        metabolites = df_reactions["metabolites"][ind]
        metabolite_list = list(metabolites.keys())
        
        for met in metabolites:
            if metabolites[met] < 0:
                substrates.append(met)
            elif metabolites[met] > 0:
                products.append(met)
                
        df_reactions["substrates"][ind] = substrates
        df_reactions["products"][ind] = products
        
    return(df_reactions)

def get_kegg_and_bigg_compound_ids(df_KM, model_metabolites, model_metabolites_list):
    df_KM["KEGG ID"] = np.nan
    df_KM['bigg.metabolite'] = np.nan
    df_KM['metanetx ID'] = np.nan
    for ind in df_KM.index:
        substrate = df_KM["substrate"][ind]
        try:
            df_KM["KEGG ID"][ind] = model_metabolites[model_metabolites_list.index(substrate)]["annotation"]["kegg.compound"][0]
        except: pass
        
        try:
            df_KM['bigg.metabolite'][ind] = model_metabolites[model_metabolites_list.index(substrate)]["annotation"]['bigg.metabolite'][0]
        except: pass
        
        try:
            df_KM['metanetx ID'][ind] = model_metabolites[model_metabolites_list.index(substrate)]["annotation"]['metanetx.chemical'][0]
        except: pass
        
    return(df_KM)

def add_substrate_name(df_KM):
    df_KM["substrate name"] = np.nan
    for ind in df_KM.index:
        bigg_metabolite = df_KM["bigg.metabolite"][ind]
        df_KM["substrate name"][ind] = bigg_metabolites[bigg_metabolites_list.index(bigg_metabolite)]["name"]
    return(df_KM)

def find_KEGG_ID_by_synonym(df_KM):
    for ind in df_KM.index:
        name = df_KM["substrate name"][ind]
        if pd.isnull(df_KM["KEGG ID"][ind]) and name is not None:
            help_df = KEGG_substrate_df.loc[KEGG_substrate_df["substrate_lower"] == name.lower()]
            if len(help_df) == 0:
                bracket_pos = name.rfind(" (")
                if bracket_pos != -1:
                    name2 = name[:bracket_pos]
                    help_df = KEGG_substrate_df.loc[KEGG_substrate_df["substrate_lower"] == name2.lower()]
                    if len(help_df) > 0:
                        df_KM["KEGG ID"][ind] = list(help_df["KEGG ID"])[0]
            else:
                df_KM["KEGG ID"][ind] = list(help_df["KEGG ID"])[0]
    return(df_KM)



def download_model_information(bigg_ID):
    r = requests.get('http://bigg.ucsd.edu/api/v2/models/' + bigg_ID + '/download')
    model = r.json() 

    model_metabolites = model["metabolites"]
    model_reactions = model["reactions"]

    df_reactions = pd.DataFrame(columns = ["BiGG reaction ID", "reaction name"])

    for reaction in model_reactions:
        df_reactions = df_reactions.append({"BiGG reaction ID" : reaction["id"],
                                                   "reaction name" : reaction["name"]}, ignore_index = True)
        
    return(model_metabolites, model_reactions, df_reactions)
    
    
def create_KM_DataFrame(df_reactions):
    df_KM = pd.DataFrame(columns = ["BiGG reaction ID", "reaction name", "gene_reaction_rule", "substrate"])
    for ind in df_reactions.index:
        substrates = []
        if df_reactions["direction"][ind] == "forward":
            substrates = substrates + df_reactions["substrates"][ind]
        elif df_reactions["direction"][ind] == "reversible":
            substrates = substrates + df_reactions["substrates"][ind] +df_reactions["products"][ind]

        for substrate in substrates:
            df_KM = df_KM.append({"BiGG reaction ID" : df_reactions["BiGG reaction ID"][ind],
                          "reaction name" : df_reactions["reaction name"][ind],
                          "gene_reaction_rule" : df_reactions["gene_reaction_rule"][ind],
                          "Uniprot ID" : df_reactions["Uniprot ID"][ind],
                          "substrate" : substrate }, ignore_index = True)
            
    
    return(df_KM)

def download_SMILES_or_KEGG_from_MetaNetX(df_KM):
    df_KM["SMILES"] = np.nan
    for ind in df_KM.index:
        if pd.isnull(df_KM["KEGG ID"][ind]) and not pd.isnull(df_KM["metanetx ID"][ind]):
            metanetx = df_KM["metanetx ID"][ind]
            reg_url = "https://www.metanetx.org/chem_info/" + metanetx
            req = Request(url=reg_url, headers=headers) 
            html = str(urlopen(req).read())

            #get SMILES:
            start_smiles = '<tr><td class="smiles">SMILES</td><td class="smiles">'
            if html.find(start_smiles) != -1:
                smiles = html[html.find(start_smiles) + len(start_smiles): ]
                smiles = smiles[ : smiles.find("</td></tr>\\n")]
                if smiles != "&nbsp;":
                    df_KM["SMILES"][ind] = smiles
                else:
                    df_KM["SMILES"][ind] = np.nan

            #get kegg ID:
            start_kegg = "https://www.kegg.jp/entry/"
            if html.find(start_kegg) != -1:
                kegg = html[html.find(start_kegg) + len(start_kegg):]
                kegg = kegg[ : kegg.find("\\")]
                df_KM["KEGG ID"][ind] = kegg
    return(df_KM)

def remove_pseudo_reactions(df_reactions):
    droplist = []
    for ind in df_reactions.index:
        if len(df_reactions["substrates"][ind]) == 0 or len(df_reactions["products"][ind]) == 0:
            droplist.append(ind)
    df_reactions.drop(droplist, inplace = True)
    return(df_reactions)

def replace_text_in_gene_reaction_rule(grr):
    grr = grr.replace("(", "")
    grr = grr.replace(")", "")
    grr = grr.replace("and", "")
    grr = grr.replace("or", "")
    return(grr)

def remove_small_mets(df_KM):
    droplist = []
    for ind in df_KM.index:
        if df_KM["KEGG ID"][ind] in ["C00001", "C00080", "C00007", "C00282"]:# hydrogen, H+, O2, H2
            droplist.append(ind)
    df_KM = df_KM.drop(droplist)
    df_KM.reset_index(drop = True, inplace = True)
    return(df_KM)

def create_text_file_with_all_gene_numbers(df_KM, model_ID):
    gene_numbers = []
    for ind in df_KM.index:
        grr = df_KM["gene_reaction_rule"][ind]
        grr = replace_text_in_gene_reaction_rule(grr)
        gene_numbers = gene_numbers + grr.split("  ")
    gene_numbers = list(set(gene_numbers))
    print(len(gene_numbers))

    f = open(join(datasets_dir, "BiGG_GSM", model_ID, "gene_numbers_" + model_ID + ".txt"),"w") 
    for ID in list(set(gene_numbers)):
        f.write(str(ID) + "\n")
    f.close()
    
    
def split_gene_reaction_rules(df_reactions):
    df_reactions_splitted = pd.DataFrame(columns = list(df_reactions.columns))

    for ind in df_reactions.index:
        grr = df_reactions["gene_reaction_rule"][ind]
        grr = grr.split(" or ")
        for genes in grr:
            df_reactions_splitted = df_reactions_splitted.append(df_reactions.loc[ind], ignore_index= True)
            last_index = list(df_reactions_splitted.index)[-1]
            df_reactions_splitted["gene_reaction_rule"][last_index] = genes
    
    return(df_reactions_splitted)

def process_reactions_DataFrame_V2(df_reactions, model_reactions, model_reactions_list):
    df_reactions = remove_non_enzymatic_reactions(df_reactions)
    df_reactions = add_reaction_information(df_reactions, model_reactions, model_reactions_list)
    df_reactions = get_substrates_and_products(df_reactions)
    df_reactions = remove_pseudo_reactions(df_reactions)
    df_reactions = split_gene_reaction_rules(df_reactions)
    df_reactions =  add_column_with_list_of_genes_V2(df_reactions)
    return(df_reactions)


def process_reactions_DataFrame(df_reactions, model_reactions, model_reactions_list):
    df_reactions = remove_non_enzymatic_reactions(df_reactions)
    df_reactions = add_reaction_information(df_reactions, model_reactions, model_reactions_list)
    df_reactions = get_substrates_and_products(df_reactions)
    df_reactions = remove_pseudo_reactions(df_reactions)
    df_reactions = split_gene_reaction_rules(df_reactions)
    df_reactions =  add_column_with_list_of_genes(df_reactions)
    return(df_reactions)

def add_column_with_list_of_genes(df_reactions):
    df_reactions["genes"] = ""
    for ind in df_reactions.index:
        grr = df_reactions["gene_reaction_rule"][ind]
        genes = replace_text_in_gene_reaction_rule(grr).split(" ")
        df_reactions["genes"][ind] = list(np.array(genes)[np.array(genes) != ""]) 
    return(df_reactions)


def add_column_with_list_of_genes_V2(df_reactions):
    df_reactions["genes"] = ""
    for ind in df_reactions.index:
        grr = df_reactions["gene_reaction_rule"][ind]
        genes = replace_text_in_gene_reaction_rule(grr).split(" ")
        genes = list(np.array(genes)[np.array(genes) != ""]) 
        genes = [gene.split("_")[0] for gene in genes]
        df_reactions["genes"][ind] = genes
    return(df_reactions)


def create_txt_file_with_all_genes(df_reactions, model_ID):
    gene_numbers = []
    for ind in df_reactions.index:
        gene_numbers = gene_numbers +  df_reactions["genes"][ind]

    f = open(join(datasets_dir, "BiGG_GSM", model_ID, "gene_numbers_" + model_ID + ".txt"),"w") 
    for ID in list(set(gene_numbers)):
        f.write(str(ID) + "\n")
    f.close()
    
    print("Txt-file with all genes is saved at: %s" % 
          join(datasets_dir, "BiGG_GSM", model_ID, "gene_numbers_" + model_ID + ".txt"))
    
def get_uids_from_gene_numbers(genes, Uniprot_df):
    UIDs = []
    for gene in genes:
        try:
            UIDs.append(list(Uniprot_df["Uniprot ID"].loc[Uniprot_df["gene number"] == gene])[0])
        except:
            UIDs.append(np.nan)
    return(UIDs)

def add_Uniprot_IDs(df_reactions, model_ID):
    Uniprot_df = pd.read_csv(join(datasets_dir, "BiGG_GSM", model_ID,
                                  "Uniprot_Mapping_" + model_ID + ".csv"), sep = ";")
    df_reactions["UIDs"] = ""
    for ind in df_reactions.index:
        genes = df_reactions["genes"][ind]
        df_reactions["UIDs"][ind] = get_uids_from_gene_numbers(genes, Uniprot_df)
        
        
    df_reactions["Uniprot ID"] = np.nan
    for ind in df_reactions.index:
        if len(df_reactions["genes"][ind]) == 1 and len(df_reactions["UIDs"][ind]) == 1:
            df_reactions["Uniprot ID"][ind] = df_reactions["UIDs"][ind][0]
        
    return(df_reactions)

def create_quick_go_link(UIDs, model_ID, part = 0):
    link = "https://www.ebi.ac.uk/QuickGO/annotations?geneProductId="
    for ID in UIDs:
        if not pd.isnull(ID):
            link = link + "UniProtKB:" + ID + ","
    link = link[:-1] + "&geneProductType=protein"
    
    #save_link
    f = open(join(datasets_dir,"BiGG_GSM", model_ID,
                  "Uniprot_IDs_enzyme_complexes_Quick_GO_" + model_ID + "part_" +str(part)+".txt"),"w")
    f.write(link)
    f.close()
    
def get_Quick_GO_links_for_enzyme_complexes(df_reactions, model_ID):
    UIDs = []
    for ind in df_reactions.index:
        if pd.isnull(df_reactions["Uniprot ID"][ind]):
            UIDs = UIDs + df_reactions["UIDs"][ind]
    UIDs = list(set(UIDs))
    
    parts = int(np.ceil(len(UIDs) / 400))
    for k in range(parts):
        create_quick_go_link(UIDs[400*k : 400*(k+1)], model_ID = model_ID, part = k)
        
def get_number_of_parts(model_ID):
    model_files = os.listdir(join(datasets_dir, "BiGG_GSM",model_ID))
    parts = []
    for file in model_files:
        pos = file.find("part_")
        if pos != -1:
            parts.append(int(file[pos+len("part_"): pos+len("part_") +1]))
        
    return(max(parts))

def load_GO_DataFrames(model_ID):
    parts = get_number_of_parts(model_ID)
    GO_df = pd.read_csv(join(datasets_dir, "BiGG_GSM",model_ID, "QuickGO-annotations-" +
                             model_ID + "part_" + str(0) +".tsv"), sep = "\t")
    if parts > 0:
        for k in range(1,parts+1):
            GO_df = GO_df.append(pd.read_csv(join(datasets_dir, "BiGG_GSM",model_ID, "QuickGO-annotations-" +
                             model_ID + "part_" + str(k) +".tsv"), sep = "\t"), ignore_index= True)
    
    GO_df = GO_df.loc[GO_df["GO ASPECT"] == "F"]
    
    droplist = []
    for ind in GO_df.index:
        go_name = GO_df["GO NAME"][ind]
        if ("binding" not in go_name or "ion" in go_name or "iron" in go_name 
            or "identical protein binding" == go_name or 'protein binding' == go_name):
            droplist.append(ind)

    GO_df.drop(droplist, inplace = True)
    
    return(GO_df)

def add_Uniprot_ID_for_enzyme_complexes(df_reactions, GO_UIDs):

    for ind in df_reactions.index:
        if pd.isnull(df_reactions["Uniprot ID"][ind]) and np.nan not in df_reactions["UIDs"][ind]:
            binding_UIDs = []
            UIDs = df_reactions["UIDs"][ind]
            for uid in UIDs:
                if uid in GO_UIDs:
                    binding_UIDs.append(uid)

            if len(binding_UIDs) == 1:
                df_reactions["Uniprot ID"][ind] = binding_UIDs[0]
    return(df_reactions)