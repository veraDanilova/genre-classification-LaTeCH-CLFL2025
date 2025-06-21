from collections import defaultdict
import pandas as pd

# map version 1

cl_map_v1 = defaultdict(dict)

cl_map_v1['academic'] = {
    'CORE' : ['RA','TR'], #technical report 'TR'
    'UDM' : ['academic'],
    'FTD' : ['A14','A15']
}

cl_map_v1['nf_prose'] = {
    'CORE' : ['EN', 'RV', 'HA', 'IB', 'OB', 'LE', 'MA', 'TB', 'RS', 'PB'],
    'UDM' : ['nonfiction_prose', 'reviews', 'blog', 'wiki'],
    'FTD' : ['A11','A16','A6','A3']
}
cl_map_v1['fiction'] = {
    'CORE' : ['OL', 'PO', 'PR', 'SL', 'SS', 'LY'],
    'UDM' : ['fiction'],
    'FTD' : ['A4','A19','A5']
}
cl_map_v1['administrative'] = {
    'CORE' : ['ED'],
    'UDM' : ['parliament'],
    'FTD' : ['A21','A13']
}
cl_map_v1['guide'] = {
    'CORE' : ['AV','TS','FH','RE','HI','HT','OH','How-to'],
    'UDM' : ['guide'],
    'FTD' : ['A7']
}
cl_map_v1['advertisement'] = {
    'CORE' : ['AD','DS'],
    'UDM' : [],
    'FTD' : ['A12']  
}
cl_map_v1['legal'] = {
    'CORE' : ['LT'],
    'UDM' : ['legal'],
    'FTD' : ['A9']
}
cl_map_v1['QA'] = {
    'CORE' : ['FI', 'QA'],
    'UDM' : ['QA'],
    'FTD' : ['A20']
}
cl_map_v1['news'] = {
    'CORE' : ['NE'],
    'UDM' : ['news'],
    'FTD' : ['A8']
}

# map version 2

cl_map_v2 = defaultdict(dict)

cl_map_v2['academic'] = {
    'CORE' : ['RA','TR'], #technical report 'TR'
    'UDM' : ['academic'],
    'FTD' : ['A14','A15']
}

cl_map_v2['nf_prose'] = {
    'CORE' : ['HA', 'PB', 'OB', 'TB', 'MA','NE'],
    'UDM' : ['nonfiction_prose', 'blog','news'],
    'FTD' : ['A11','A3','A8']
}

cl_map_v2['fiction'] = {
    'CORE' : ['OL', 'PO', 'PR', 'SL', 'SS', 'LY'],
    'UDM' : ['fiction'],
    'FTD' : ['A4','A19']
}
cl_map_v2['administrative'] = {
    'CORE' : ['LT'],
    'UDM' : ['parliament','legal'],
    'FTD' : ['A21','A13','A9','A20']
}
cl_map_v2['guide'] = {
    'CORE' : ['HT','RE','AV','TS','OH','HI','How-to','FI', 'QA', 'FH'],
    'UDM' : ['guide','QA'],
    'FTD' : ['A7']
}
cl_map_v2['advertisement'] = {
    'CORE' : ['AD','DS'],
    'UDM' : [],
    'FTD' : ['A12']  
}

# map version 3

cl_map_v3 = defaultdict(dict)

cl_map_v3['academic'] = {
    'CORE' : ['RA', 'TR'], #technical report 'TR'
    'UDM' : ['academic'],
    'FTD' : ['A14','A15']
}

cl_map_v3['nf_prose'] = {
    'CORE' : ['HA', 'PB', 'OB', 'TB', 'MA'],
    'UDM' : ['nonfiction_prose', 'blog'],
    'FTD' : ['A11','A3']
}
cl_map_v3['fiction'] = {
    'CORE' : ['OL', 'PO', 'PR', 'SL', 'SS', 'LY'],
    'UDM' : ['fiction'],
    'FTD' : ['A4','A19']
}
cl_map_v3['administrative'] = {
    'CORE' : [],
    'UDM' : ['parliament'],
    'FTD' : ['A21','A13','A20']
}
cl_map_v3['guide'] = {
    'CORE' : ['HT','RE','AV','TS','OH','HI','How-to'],
    'UDM' : ['guide'],
    'FTD' : ['A7']
}
cl_map_v3['advertisement'] = {
    'CORE' : ['AD','DS'],
    'UDM' : [],
    'FTD' : ['A12']  
}
cl_map_v3['legal'] = {
    'CORE' : ['LT'],
    'UDM' : ['legal'],
    'FTD' : ['A9']
}
cl_map_v3['QA'] = {
    'CORE' : ['FI', 'QA', 'FH'],#
    'UDM' : ['QA'],
    'FTD' : ['A20']
}
cl_map_v3['news'] = {
    'CORE' : ['NE'],
    'UDM' : ['news'],
    'FTD' : ['A8']
}

maps = {"cl_map_v1": cl_map_v1,
        "cl_map_v2": cl_map_v2,
        "cl_map_v3": cl_map_v3}

language_families = {'Afro-Asiatic': ['Hebrew', 'Maltese'],
'Altaic': ['Turkish', 'Uyghur'],
'Austronesian': ['Indonesian'],
'IE.Celtic': ['Scottish Gaelic'],
'Code-switch':['Hindi English', 'Turkish German'],
'Creole': ['Naija'],
'IE.Baltic': ['Lithuanian'], 
'IE.Germanic': ['Afrikaans', 'Dutch', 'English', 'German', 'Icelandic', 'Norwegian', 'Swedish'],
'IE.Greek': ['Greek'],
'IE.Armenian': ['Armenian', 'Western Armenian'],
'IE.Indic': ['Hindi'], 
'IE.Romance': ['Catalan', 'French', 'Italian', 'Portuguese', 'Romanian', 'Spanish'], 
'IE.Slavic': ['Belarusian', 'Bulgarian', 'Croatian', 'Czech', 'Polish', 'Russian', 'Slovak', 'Slovenian'],  
'Sino-Tibetan': ['Chinese'],
'Uralic': ['Estonian', 'Finnish', 'Erzya'],
'Dravidian':['Tamil']
}

organizations = pd.DataFrame([
  {
      "language": "German",
      "country": "DE",
      "disease": "Allergy, Asthma",
      "organization": "Heufieberbund von Helgoland Deutscher Allergiker Bund Deutscher Allergiker und Asthmatiker Bund",
      "publications": [
          "Jahresberichte",
          "Der Allergiker",
          "Der Allergiker und Asthmatiker"
      ],
      "folders": ["DAAB", "der_allergiker",  "jahresberichte"]
  },
  {
      "language": "German",
      "country": "DE",
      "disease": "Diabetes",
      "organization": "Deutscher Diabetiker Bund",
      "publications": [
          "Wir Zuckerkranken",
          "Der Diabetiker",
          "Diabetes Journal"
      ],
      "folders": ["diabetiker_journal", "wir_zuckerkranken"]
  },
  {
      "language": "German",
      "country": "DE",
      "disease": "Multiple Sclerosis",
      "organization": "Deutsche Multipel Sklerose Gesellschaft",
      "publications": [
          "Aus unserer Arbeit",
          "Bericht aus der Arbeit",
          "Mitteilungsblatt",
          "DMSG Aktiv"
      ],
      "folders": ["ms_mitteilungsblatt"]
  },
  {
      "language": "Swedish",
      "country": "SE",
      "disease": "Allergy, Asthma",
      "organization": "Astma- och allergiförbundet / Riksförbundet mot astma och andra allergiska sjukdomar",
      "publications": ["Allergia"],
      "folders": ["allergia"]
  },
  {
      "language": "Swedish",
      "country": "SE",
      "disease": "Heart and Lung Disease",
      "organization": "De lungsjukas eftervårdskommitté / Hjärt- och lungsjukas riksförbund / Riksförbundet HjärtLung",
      "publications": ["Status"],
      "folders": ["status"]
  },
  {
      "language": "Swedish",
      "country": "SE",
      "disease": "Diabetes",
      "organization": "Riksförbundet för sockersjuka / Svenska Diabetesförbundet",
      "publications": ["Diabetes"],
      "folders": ["diabetes"]
  },
  {
      "language": "French",
      "country": "FR",
      "disease": "Paralysis, Rheumatism, Multiple Sclerosis",
      "organization": "Association des paralysés et rhumatisants",
      "publications": [
          "Faire face",
          "Le Courrier de la sclérose en plaques",
          "APF informations"
      ],
      "folders": ["APF"]
  },
  {
      "language": "French",
      "country": "FR",
      "disease": "Diabetes",
      "organization": "Association Française des Diabétiques",
      "publications": [
          "Le Journal des diabétiques",
          "Le Journal de l'AFD",
          "Equilibre"
      ],
      "folders": ["ADF"]
  },
  {
      "language": "English",
      "country": "GB",
      "disease": "Diabetes",
      "organization": "British Diabetic Association",
      "publications": [
          "The Diabetic Journal",
          "Balance"
      ],
      "folders": ["BDA"]
  },
  {
      "language": "English",
      "country": "GB",
      "disease": "Polio",
      "organization": "British Polio Fellowship",
      "publications": [
          "IPF Bulletin",
          "Bulletin of the British Polio Fellowship"
      ]
  },
  {
      "language": "English",
      "country": "GB",
      "disease": "Rheumatism, Arthritis",
      "organization": "British Rheumatism and Arthritis Association",
      "publications": [
          "BRA Review",
          "Arthritis News"
      ],
      "folders": ["BRA"]
  },
  {
      "language": "Swedish",
      "country": "SE",
      "disease": None,
      "organization": None,
      "publications": ["Svenska Läkartidningen"],
      "folders": ["svenska_lakartidningen"]
  },
  {
      "language": "Swedish",
      "country": "SE",
      "disease": None,
      "organization": None,
      "publications": ["Socialmedicinsk Tidskrift"],
      "folders": ["socialmedicinsk_tidskrift"]
  },
  {
      "language": "German",
      "country": "DE",
      "disease": None,
      "organization": None,
      "publications": ["Deutsche Medizinische Wochenschrift (DMW)"],
      "folders": ["DMW"]
  },

])

wtp_language_codes = {
    "German" : "de",
    "Swedish": "sv",
    "French": "fr",
    "English": "en"
}