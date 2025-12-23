DATASET_COLUMN = {
                  'Mutagenicity':['Mutagenicity'], 
                  'hERG':['hERG'], 
                  'BBBP':['BBBP'],
                  'Lipophilicity':['Lipophilicity'],
                  'tox21': ['NR-AR', 'NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD', 
                            'NR-PPAR-gamma', 'SR-ARE','SR-ATAD5', 'SR-HSE','SR-MMP','SR-p53'],
                  'esol':['measured log solubility in mols per litre'],
                  'Benzene':['label'],
                  'Alkane_Carbonyl':['label'],
                  'Fluoride_Carbonyl':['label'],
                }

DATASET_TYPE = {
                  'Mutagenicity':'BinaryClass', 
                  'hERG':'BinaryClass', 
                  'BBBP':'BinaryClass',
                  'Lipophilicity':'Regression',
                  'tox21': 'MultiTask',
                  'esol':'Regression',
                  'Benzene':'BinaryClass',
                  'Alkane_Carbonyl':'BinaryClass',
                  'Fluoride_Carbonyl':'BinaryClass',
                }

CHOSEN_THRESHOLD = {'RBRICS': 
                    {'Mutagenicity':0.2,
                     'hERG':0.5,
                     'BBBP':0.6,
                     'Benzene':0.5,
                     'Alkane_Carbonyl':0.5,
                     'Fluoride_Carbonyl':0.5,
                     'esol':0.2,
                     'Lipophilicity':0.5,
                     'tox21':0.2},
                    'PRESERVE_ALKANE_CARBONYL': 
                    {
                     'Alkane_Carbonyl':0.5
                    }
                   }

ALGORITHMS = ["RBRICS"]#["RBRICS","BRICS"]

PERCENT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]