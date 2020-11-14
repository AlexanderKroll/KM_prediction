import numpy as np
import pandas as pd
import time
import requests
from directory_infomation import *


def download_data_from_BRENDA(function, parameters, error_identifier, print_error=False):
    """
    A function that downloads data from BRENDA. The data to be downloaded is specified by the arguments "function" and
    "parameters". The arguments "error_identifier" and "print_error" can be used for error identfication.
    """
    try:
        resultString = function(*parameters)
        return(resultString)
    except:
        #if download was unsuccessfull, try again one more time:
        time.sleep(0.5)
        try:
            resultString = function(*parameters)
            return(resultString)
        except Exception as ex:
            if print_error:
                print("Download for %s was unsuccessfull (Type of error: %s)" % (error_identifier, ex))
                return([]) # return empty list if download was unsuccessfull
    


def extract_KM_info_from_resultString(brenda_df, resultString):
    """
    Takes as an input a so called "resultString", which is produced by the function "download_data_from_BRENDA",
    and as a second input a DataFrame "brenda_df" with the columns "KM", "substrat", "ecNumber", "organism", and "commentary".
    Information about these columns is extracted from the resultString and stored in the given DataFrame brenda_df. 
    """
    if not resultString is None:
        for result in resultString:
            KM, substrate = result["kmValue"], result["substrate"] 
            if not KM == "-999": #BRENDA uses -999 for missing values
                if not substrate == "more": #We only store the data point if a specific substrate is given
                    brenda_df = brenda_df.append(other = [np.nan], ignore_index = True)
                    n = len(brenda_df) -1
                    brenda_df["KM"][n], brenda_df["substrate"][n] = KM, substrate
                    brenda_df["ecNumber"][n] = result["ecNumber"]
                    brenda_df["organism"][n] = result["organism"]
                    brenda_df["commentary"][n] = result["commentary"]
    return(brenda_df)


def substrate_names_to_Pubchem_CIDs(metabolites):
    """
    A function that maps a list of metabolites to PubChem Compound IDs (CIDs), if there is an excat match
    for the metabolite and a synonym from the Pubchem synonym list.
    """    
    
    n = len(metabolites)
    match = [np.nan] * n

    for k in range(5):
        print("loading part %s of 5 of the synonym list..." %(k+1))
        df = pd.read_pickle(datasets_dir + "substrates_synonyms_part"+str(k)+".pkl")
        substrates = list(df["substrates"])
        cid = list(df["CID"])
        df = None
        print("searching in synoynm list part %s for matches" %(k+1))
        
        for i in range(n):
            if pd.isnull(match[i]):
                met = metabolites[i].lower()
                if not pd.isnull(met):
                    try:
                        pos = substrates.index(met.lower())
                        match[i] = cid[pos]
                    except ValueError:
                        None
    df = pd.DataFrame(data= {"Metabolite" : metabolites, "CID" : match})
    return(df)


def map_BRENDA_entry_to_KEGG_reaction_ID(entry, KEGG_reaction_df):
    '''
    Using information about the EC number and the KEGG CID, a data point is mapped to all possible KEGG reaction IDs.
    '''
    #get ec_number and KEGG ID of substrate
    ec = entry["ecNumber"]
    KEGG_ID = entry["KEGG ID"]
    #save all reaction IDs from KEGG with direction in the following list:
    reaction_ids = []
    #only search if a KEGG ID was found
    if not pd.isnull(KEGG_ID):
        #takesubset of KEGG database with reactions with fitting EC number
        reaction_df = KEGG_reaction_df.loc[KEGG_reaction_df["EC number"] == ec]
        #iterate overall entries with fitting EC numbers
        for k in reaction_df.index:
            reaction_entry = reaction_df.loc[k]
            #get KEGG IDs of substrates on left and right side of reaction equation:
            left = reaction_entry["KEGG IDs left"]
            right = reaction_entry["KEGG IDs right"]
            if KEGG_ID in left:
                reaction_ids.append(reaction_entry["KEGG reaction ID"] + "_f")
            if KEGG_ID in right:
                reaction_ids.append(reaction_entry["KEGG reaction ID"] + "_b")
                
    if reaction_ids == []:
        return(None)
    else:
        return(reaction_ids)
    
    
def replace_tabs_with_spaces_in_txt_file(path_to_file):
    file_object  = open(path_to_file, "r")
    new_content = file_object.read().replace("\t", " ")
    file_object.close()
    file_object  = open(path_to_file, "w")
    file_object.write(new_content)
    file_object.close()
    

def calculate_FunD_vectors_from_Pfam_output(df, no_of_parts):
    '''
    Using the output of the Pfam webservice, this function calculates binary vectors that store the information
    about the functional domains that an enzyme contains.
    '''
    count = 0
    FUND = np.zeros((no_of_parts*500,19000))
    for part in range(no_of_parts):
        replace_tabs_with_spaces_in_txt_file(path_to_file = datasets_dir + "pfam_output_part" + str(part))
        file_object  = open(datasets_dir + "pfam_output_part" + str(part), "r")
        old_ind = np.inf
        for line in file_object:
            ind = int(line[1:line.find(" ")])
            if ind != old_ind: #new enzyme organism combination
                if old_ind != np.inf:
                    df["FunD"][old_ind] = FunD
                    FUND[count,:] = FunD
                    count +=1
                FunD = np.zeros(19000)
            PF_ID = int(line[(line.find("PF")+2) : line.find(".")])
            FunD[PF_ID] = 1
            old_ind = ind
        df["FunD"][old_ind] = FunD
    return(df, FUND)


def create_df_with_KEGG_CIDs_and_correpsonding_names():
    name_finder = '<nobr>Name</nobr></th>\n<td class="td21" style="border-color:#000; border-width: 1px 1px 0px 1px; border-style: solid"><div style="width:555px;overflow-x:auto;overflow-y:hidden"><div style="width:555px;overflow-x:auto;overflow-y:hidden">'
    name_end = '<br>\n</div></div>'
    b = len(name_finder)

    first_start = False
    CID = []
    substrate_name = []
    start = 0 

    for i in range(start,25000):
        kegg_id ="C" + (5 - len(str(i)))*"0" + str(i)
        #download html source code from kegg for this CID:
        page_source =urllib2.urlopen("https://www.genome.jp/dbget-bin/www_bget?" + kegg_id).read()
        #find substrate names for this KEGG ID:
        start_pos = page_source.find(name_finder)
        if start_pos == -1:
            print("No entry found for KEGG ID %s" % kegg_id)
        else:
            text = page_source[(start_pos + b) :]
            end_pos = text.find(name_end)
            #text now conatins all substrate names for one ID in one string. We split it into
            #mulitiple strings, each with one substrate name
            names = text[:end_pos].split(";<br>\n")
            for name in names:
                CID.append(kegg_id)
                substrate_name.append(name)

    substrate_df = pd.DataFrame(data = {"KEGG ID" : CID, "substrate" : substrate_name})
    substrate_df.to_pickle(datasets_dir + "KEGG_substrate_df.pkl")
    
    
def download_mol_files():
    """
    This function downloads all available MDL Molfiles for alle substrate with a KEGG Compound ID between 0 and 22500.    
    """
    #create folder where mol-files shalle be stored:
    try:
        os.mkdir(datasets_dir + "mol-files/")
    except:
        print("Folder for mol-files already exitsts. If you want to download all mol-files again, first remove the current folder.")
        return None
    #Download all mol-files for KEGG IDs betweeen 0 and 22500
    for i in range(0,22500):
        print(i)
        kegg_id = "C" +(5-len(str(i)))*"0" + str(i)
        #get mol-file:
        r = requests.get(url = "https://www.genome.jp/dbget-bin/www_bget?-f+m+compound+"+kegg_id)
        #check if it's empty
        if not r.content == b'':
            f= open(datasets_dir + "mol-files/" +kegg_id + ".mol","wb")
            f.write(r.content)
            f.close()