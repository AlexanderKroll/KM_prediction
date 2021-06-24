import numpy as np
import pandas as pd
import time
import pickle

from zeep import Client
import hashlib
import requests
from os.path import join
from urllib.request import urlopen, Request
from ete3 import NCBITaxa
ncbi = NCBITaxa()

from directory_infomation import *
from functions_and_dicts_data_preprocessing_GNN import *


headers= {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
                      'AppleWebKit/537.11 (KHTML, like Gecko) '
                      'Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}


wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
password = hashlib.sha256("a2b8c6".encode("utf-8")).hexdigest()
email = "alexander.kroll@hhu.de"
client = Client(wsdl)



def array_column_to_strings(df, column):
    df[column] = [str(list(df[column][ind])) for ind in df.index]
    return(df)

def string_column_to_array(df, column):
    df[column] = [np.array(eval(df[column][ind])) for ind in df.index]
    return(df)

def download_KM_from_Brenda(EC):
    reg_url = "https://www.brenda-enzymes.org/enzyme.php?ecno=" + EC
    req = Request(url=reg_url, headers=headers) 
    html = str(urlopen(req).read())
    html = html[html.find('<a name="KM VALUE [mM]"></a>') : ]
    html = html[ : html.find(" </div>\\n")]
    return(html)

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


def get_entry_from_html(html, entry, sub_entry = 0):
    ID = "tab12r" + str(entry) + "sr" + str(sub_entry) + "c"
    data = []
    for i in range(6):
        search_string = '<div id="'+ ID + str(i) + '" class="cell"><span>'
        pos = html.find(search_string) 
        if pos == -1: #string was not found, try again with different string
            search_string = '<div id="'+ ID + str(i) + '" class="cell notopborder">'
            pos = html.find(search_string)
            if pos == -1: #string was not found, try again with different string
                search_string = '<div id="'+ ID + str(i) + '" class="cell"><span'
                pos = html.find(search_string)
        if pos != -1: #string was found
            subtext = html[pos+len(search_string):]
            data.append(subtext[:subtext.find("\\n")])
        else: 
            return([])
    return(data)

def process_string(string):
    if string[0] == "<":
        string = string[string.find(">")+1:]
    string = string.replace("\\", "")
    string = string.replace("</a>", "")
    return(string)


def process_string_V2(string):
    if ">" in string:
        string = string[string.find(">")+1:]
    return(string)


def process_UNIPROT_string(string):
    if "</span></div><div id=" in string:
        string = string[:string.find("</span></div><div id=")]
    return(string)


def process_UNIPROT_ID(string):
    if string[0] == "<" or string[1] == "<":
        string = string[string.find(">")+1:]
    return(string)


def add_KM_for_EC_number(brenda_df, EC):
    html = download_KM_from_Brenda(EC = EC)
    
    entry = 0
    sub_entry = 0
    found_entry = True
    while found_entry == True:

        data = get_entry_from_html(html = html, entry = entry, sub_entry = sub_entry)
        if data != []:
            found_entry = True
            sub_entry +=1
            
            KM =  process_string(data[0])
            UNIPROT = process_string(data[3])
            
            if UNIPROT != "-":
                UNIPROT = process_UNIPROT_string(UNIPROT)
                UNIPROT_list = [process_UNIPROT_ID(ID) for ID in  UNIPROT.split(", ")]
            else:
                UNIPROT_list = []
                
            if "additional information" not in KM:
                brenda_df = brenda_df.append({"EC": EC, "KM VALUE" : KM, 
                                              "SUBSTRATE": process_string_V2(process_string(data[1])),
                                              "ORGANISM": process_string_V2(process_string(data[2])),
                                              "UNIPROT": UNIPROT_list,
                                              "COMMENTARY": process_string(data[4]), "LITERATURE": process_string(data[5])},
                                              ignore_index= True)
        elif sub_entry == 0:
            found_entry = False
        else:
            entry +=1
            sub_entry = 0
    return(brenda_df)


def is_bacteria(org):
    try:
        tax_id = ncbi.get_name_translator([org])[org][0]
        lineage = ncbi.get_lineage(tax_id)
        if 2 not in lineage:
            return(False)
        else:
            return(True)
    except KeyError:
        return(False)


def substrate_names_to_Pubchem_CIDs(metabolites):
    """
    A function that maps a list of metabolites to PubChem Compound IDs (CIDs), if there is an excat match
    for the metabolite and a synonym from the Pubchem synonym list.
    """    
    
    n = len(metabolites)
    match = [np.nan] * n

    for k in range(5):
        print("loading part %s of 5 of the synonym list..." %(k+1))
        df = pd.read_pickle(join(datasets_dir, "substrate_synonyms", "substrates_synonyms_part"+str(k)+".pkl"))
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
    ec = entry["EC"]
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
    substrate_df.to_pickle(datasets_dir + "\\substrate_synonyms\\KEGG_substrate_df.pkl")
    
    
def create_df_with_KEGG_drug_IDs_and_correpsonding_names():
    name_finder = '<tr><th class="th51" align="left" valign="top" style="border-color:#000; border-width: 1px 0px 0px 1px; border-style: solid"><nobr>Name</nobr></th>\n<td class="td51" style="border-color:#000; border-width: 1px 1px 0px 1px; border-style: solid"><div style="width:555px;overflow-x:auto;overflow-y:hidden">'
    name_end = '</div></td></tr>'
    b = len(name_finder)

    first_start = False
    CID = []
    substrate_name = []
    start = 0 

    for i in range(start,12500):
        kegg_id ="D" + (5 - len(str(i)))*"0" + str(i)
        #download html source code from kegg for this CID:
        page_source = requests.get("https://www.genome.jp/dbget-bin/www_bget?" + kegg_id).text
        #find substrate names for this KEGG ID:
        start_pos = page_source.find(name_finder)
        if start_pos == -1:
            print("No entry found for KEGG ID %s" % kegg_id)
        else:
            text = page_source[(start_pos + b) :]
            end_pos = text.find(name_end)
            #text now conatins all substrate names for one ID in one string. We split it into
            #mulitiple strings, each with one substrate name
            names = text[:end_pos].split(";<br>")
            for name in names:
                CID.append(kegg_id)
                substrate_name.append(name)

    durgs_df = pd.DataFrame(data = {"KEGG ID" : CID, "substrate" : substrate_name})
    
    drugs_df = pd.read_pickle(datasets_dir + "KEGG_drugs_df.pkl")

    for ind in drugs_df.index:
        drug_ID = drugs_df["KEGG ID"][ind]
        drug_name = drugs_df["substrate"][ind]
        end_pos = drug_name.rfind("(")
        if end_pos != -1:
            drugs_df = drugs_df.append({"KEGG ID" : drug_ID, "substrate" : drug_name[:end_pos-1]}, ignore_index = True)
    
    durgs_df.to_pickle(datasets_dir + "\\substrate_synonyms\\KEGG_drugs_df.pkl")
    
    
def replace_ranges_of_KM_values_with_means(df):
    for ind in df.index:
        KM = df['KM VALUE'][ind]
        if KM.find("-") != -1:
            KM = KM.split(" - ")
            KM = np.mean([float(km) for km in KM])
            
        df['KM VALUE'][ind] = float(KM)
    return(df)
    
    
def download_mol_files():
    """
    This function downloads all available MDL Molfiles for alle substrate with a KEGG Compound ID between 0 and 22500.    
    """
    #create folder where mol-files shalle be stored:
    try:
        os.mkdir(join(datasets_dir, "mol-files/"))
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
            f= open(join(datasets_dir, "mol-files",  kegg_id + ".mol"),"wb")
            f.write(r.content)
            f.close()
            
    for i in range(0,12000):
        print(i)
        kegg_id = "D" +(5-len(str(i)))*"0" + str(i)
        #get mol-file:
        r = requests.get(url = "https://www.genome.jp/dbget-bin/www_bget?-f+m+compound+"+kegg_id)
        #check if it's empty
        if not r.content == b'':
            f= open(join(datasets_dir, "mol-files",  kegg_id + ".mol"),"wb")
            f.write(r.content)
            f.close()
            
def add_Unirep_vector(df, Unirep_df):
    X_Unirep = Unirep_df.values
    df["Unirep"] = ""
    for i in range(len(X_Unirep)):
        ind = X_Unirep[i,0]
        df["Unirep"][ind] = X_Unirep[i,1:]
    return(df)


def get_PMIDs_that_are_in_BRENDA(Sabio_PMIDs, df):
    PMIDs_in_BRENDA = []

    for PMID in Sabio_PMIDs:

        help_df = df.loc[df["PubMedID"] == PMID]
        EC = list(help_df["ECNumber"])[0]

        parameters = (email, password,"ecNumber*" + EC, "reference*", "authors*", "title*", "journal*",
                      "volume*", "pages*", "year*", "organism*", "commentary*", "pubmedId*"+ str(int(PMID)), "textmining*")
        resultString1 = client.service.getReference(*parameters)   

        if len(resultString1) > 0:
            for i in range(len(resultString1)):
                reference = resultString1[i]["reference"]
                organism = resultString1[i]["organism"]
                ecNumber = resultString1[i]["ecNumber"]

                parameters = (email ,password,"ecNumber*" +ecNumber, "kmValue*", "kmValueMaximum*", "substrate*", "commentary*", "organism*", "ligandStructureId*", "literature*" +reference)
                resultString2 = client.service.getKmValue(*parameters)
                if len(resultString2) > 0:
                    PMIDs_in_BRENDA.append(PMID)
                    break
                time.sleep(0.1)
        time.sleep(0.1)
        
    return(PMIDs_in_BRENDA)


def download_sabio_data():
    """function to download all availiable data from the sabio database
    Collects all Entry IDs and downloads the requested columns for each Entry ID

    Returns
    -------
    DataFrame
        the downloaded data

    """
    # ------------------------------------------------------------------------------
    # SABIO-RK data script (taken from sabiork.h-its.org and adjusted to my needs)
    # ------------------------------------------------------------------------------

    ENTRYID_QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/entryIDs'
    PARAM_QUERY_URL = 'http://sabiork.h-its.org/entry/exportToExcelCustomizable'

    # ask SABIO-RK for all EntryIDs and create a query string
    query_dict = {'EntryID': '*'}
    query_string = ' OR '.join(['%s:%s' % (k, v) for k, v in query_dict.items()])
    query = {'format': 'txt', 'q': query_string}

    # make GET request
    request = requests.get(ENTRYID_QUERY_URL, params=query)
    request.raise_for_status()  # raise if 404 error

    # each entry is reported on a new line
    entryIDs = [int(x) for x in request.text.strip().split('\n')]
    print('%d matching entries found.' % len(entryIDs))

    # encode next request, for parameter data given entry IDs
    data_field = {'entryIDs[]': entryIDs}
    query = {'format': 'tsv', 'fields[]': ['EntryID', 'Organism', 'UniprotID',
                                           'ECNumber', "Pathway", 'Parameter',
                                           'EnzymeType', 'temperature', "pH",
                                           "Enzymename", "KeggReactionID",
                                           "PubMedID", "Substrate",
                                           "Product", "Inhibitor", "Cofactor",
                                           "Activator", "PubChemID", "Tissue",
                                           "CellularLocation"]}
    # not working: Catalyst, OtherModifier, AnyRole

    # make POST request
    print("Downloading...")
    request = requests.post(PARAM_QUERY_URL, params=query, data=data_field)
    request.raise_for_status()
    print("Download finished!", len(request.text.split("\n")))

    # results
    results = [el.split("\t") for el in request.text.split("\n")]
    headers = results.pop(0)

    # remove last line which only consists of empty values
    del results[-1]

    df = pd.DataFrame(results, columns=headers)

    return df

def preprocess_raw_sabio_DataFrame(df):
    #only keep data points with Km values (that ar nonzero):
    df = df.loc[df["parameter.type"] == "Km"]
    df = df.loc[df["parameter.startValue"] != 0]
    #remove all data points without Uniprot IDs
    df = df.loc[df["UniprotID"] != ""]
    df = df.loc[~pd.isnull(df["UniprotID"])]
    #only keep entries with unit Mol:
    df = df.loc[df["parameter.unit"] == "M"]

    #only keep entries with a PubMedID:
    df = df.loc[~pd.isnull(df["PubMedID"])]

    df = df.loc[df["EnzymeType"] == "wildtype"]
    #only keep entries with a KeGG reaction ID

    df = df.loc[~pd.isnull(df["KeggReactionID"])]
    
    
    #only keep necessary columns:
    df = pd.DataFrame(data = {"Organism" : df["Organism"], "ECNumber" : df["ECNumber"], "KM" : df["parameter.startValue"],
                               "PubMedID" : df["PubMedID"], "substrate" : df["parameter.associatedSpecies"],
                               "UniprotID" : df["UniprotID"]})
    
    df["log10_KM"] = [np.log10(Km*10**3) for Km in df["KM"]]
    
    #removing unrealsitc outliers:
    df = df.loc[df["log10_KM"]> -5]
    df = df.loc[df["log10_KM"]< 4]
    
    return(df)

pca = pickle.load(open(join(datasets_dir, "enzyme_data", "PCA_Unirep.pkl"),'rb'))
mean = np.load(join(datasets_dir, "enzyme_data", "mean_PCA_Unirep.npy"))
std = np.load(join(datasets_dir, "enzyme_data", "std_PCA_Unirep.npy"))


def calculate_and_save_input_matrixes(sample_ID, molecule_ID, unirep, extras,
                                      save_folder = join(datasets_dir, "GNN_input_data")):
        
    unirep20 = pca.transform(np.array([unirep]))[0]
    unirep20 = (unirep20 - mean) / std
    
    
    
    [XE, X, A] = create_input_data_for_GNN_for_substrates(substrate_ID = molecule_ID, print_error = True)
    if not A is None:
        np.save(join(save_folder, sample_ID + '_X.npy'), X) #feature matrix of atoms/nodes
        np.save(join(save_folder, sample_ID + '_XE.npy'), XE) #feature matrix of atoms/nodes and bonds/edges
        np.save(join(save_folder, sample_ID + '_A.npy'), A) 
        np.save(join(save_folder, sample_ID + '_Unirep20.npy'), unirep20)
        np.save(join(save_folder, sample_ID + '_extras.npy'), extras)
        return(-1)
    else:
        return(int(sample_ID.split("_")[1]))
    
    
input_data_folder = join(datasets_dir, "GNN_input_data")

def get_representation_input(cid_list):
    XE = ();
    X = ();
    A = ();
    UniRep = ();
    extras = ();
    # Generate data
    for cid in cid_list:

        X = X + (np.load(join(input_data_folder, cid + '_X.npy')), );
        XE = XE + (np.load(join(input_data_folder, cid + '_XE.npy')), );
        A = A + (np.load(join(input_data_folder, cid + '_A.npy')), );
        extras =  extras + (np.load(join(input_data_folder, cid + '_extras.npy')), );
            
    return(XE, X, A, extras)


def get_substrate_representations(df, prefix):
    df["GNN FP"] = ""
    i = 0
    n = len(df)
    UniRep = np.zeros((64, 20))
    
    cid_all = list(df.index)
    cid_all = [prefix + str(cid) for cid in cid_all]
    
    while i*64 <= n:
        if (i+1)*64  <= n:
            XE, X, A, extras = get_representation_input(cid_all[i*64:(i+1)*64])
            representations = get_fingerprint_fct([np.array(XE), np.array(X),np.array(A), np.array(UniRep),
                                                   np.array(extras)])[0]
            df["GNN FP"][i*64:(i+1)*64] = list(representations[:, :52])
        else:
            print(i)
            XE, X, A, extras = get_representation_input(cid_all[-64:])
            representations = get_fingerprint_fct([np.array(XE), np.array(X),np.array(A), np.array(UniRep),
                                                   np.array(extras)])[0]
            df["GNN FP"][-64:] = list(representations[:, :52])
        i += 1
        
    ### set all GNN FP-entries with no input matrices to np.nan:
    all_X_matrices = os.listdir(input_data_folder)
    for ind in df.index:
        if prefix +str(ind) +"_X.npy" not in all_X_matrices:
            df["GNN FP"][ind] = np.nan
    return(df)
