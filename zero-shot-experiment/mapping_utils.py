"""
Label mappings from source-dataset labels to unified genre labels,
and language-family groupings.

Three map versions reflect different granularity / grouping decisions:
  cl_map_v1 – finest-grained (9 classes, separate news/QA/legal)
  cl_map_v2 – merged news+nonfiction, merged QA+guide+legal
  cl_map_v3 – same 9 classes as v1 but slightly different CORE/FTD assignments
"""

from collections import defaultdict

# ---------------------------------------------------------------------------
# Map version 1  –  9 classes
# ---------------------------------------------------------------------------
cl_map_v1 = defaultdict(dict)
cl_map_v1['academic']       = {'CORE': ['RA','TR'],                                                   'UDM': ['academic'],                            'FTD': ['A14','A15']}
cl_map_v1['nonfiction']     = {'CORE': ['EN','RV','HA','IB','OB','LE','MA','TB','RS','PB'],            'UDM': ['nonfiction_prose','reviews','blog','wiki'],'FTD': ['A11','A16','A6','A3']}
cl_map_v1['fiction']        = {'CORE': ['OL','PO','PR','SL','SS','LY'],                               'UDM': ['fiction'],                             'FTD': ['A4','A19','A5']}
cl_map_v1['administrative'] = {'CORE': ['ED'],                                                         'UDM': ['parliament'],                          'FTD': ['A21','A13']}
cl_map_v1['guide']          = {'CORE': ['AV','TS','FH','RE','HI','HT','OH','How-to'],                  'UDM': ['guide'],                               'FTD': ['A7']}
cl_map_v1['advertisement']  = {'CORE': ['AD','DS'],                                                    'UDM': [],                                      'FTD': ['A12']}
cl_map_v1['legal']          = {'CORE': ['LT'],                                                         'UDM': ['legal'],                               'FTD': ['A9']}
cl_map_v1['QA']             = {'CORE': ['FI','QA'],                                                    'UDM': ['QA'],                                  'FTD': ['A20']}
cl_map_v1['news']           = {'CORE': ['NE'],                                                         'UDM': ['news'],                                'FTD': ['A8']}

# ---------------------------------------------------------------------------
# Map version 2  –  6 classes (news/QA/legal merged into broader categories)
# ---------------------------------------------------------------------------
cl_map_v2 = defaultdict(dict)
cl_map_v2['academic']       = {'CORE': ['RA','TR'],                                                    'UDM': ['academic'],                            'FTD': ['A14','A15']}
cl_map_v2['nonfiction']     = {'CORE': ['HA','PB','OB','TB','MA','NE'],                                'UDM': ['nonfiction_prose','blog','news'],        'FTD': ['A11','A3','A8']}
cl_map_v2['fiction']        = {'CORE': ['OL','PO','PR','SL','SS','LY'],                               'UDM': ['fiction'],                             'FTD': ['A4','A19']}
cl_map_v2['administrative'] = {'CORE': ['LT'],                                                         'UDM': ['parliament','legal'],                  'FTD': ['A21','A13','A9','A20']}
cl_map_v2['guide']          = {'CORE': ['HT','RE','AV','TS','OH','HI','How-to','FI','QA','FH'],        'UDM': ['guide','QA'],                          'FTD': ['A7']}
cl_map_v2['advertisement']  = {'CORE': ['AD','DS'],                                                    'UDM': [],                                      'FTD': ['A12']}

# ---------------------------------------------------------------------------
# Map version 3  –  9 classes (alternative CORE/FTD assignments for QA/news)
# ---------------------------------------------------------------------------
cl_map_v3 = defaultdict(dict)
cl_map_v3['academic']       = {'CORE': ['RA','TR'],                                                    'UDM': ['academic'],                            'FTD': ['A14','A15']}
cl_map_v3['nonfiction']     = {'CORE': ['HA','PB','OB','TB','MA'],                                     'UDM': ['nonfiction_prose','blog'],             'FTD': ['A11','A3']}
cl_map_v3['fiction']        = {'CORE': ['OL','PO','PR','SL','SS','LY'],                               'UDM': ['fiction'],                             'FTD': ['A4','A19']}
cl_map_v3['administrative'] = {'CORE': [],                                                              'UDM': ['parliament'],                          'FTD': ['A21','A13','A20']}
cl_map_v3['guide']          = {'CORE': ['HT','RE','AV','TS','OH','HI','How-to'],                       'UDM': ['guide'],                               'FTD': ['A7']}
cl_map_v3['advertisement']  = {'CORE': ['AD','DS'],                                                    'UDM': [],                                      'FTD': ['A12']}
cl_map_v3['legal']          = {'CORE': ['LT'],                                                         'UDM': ['legal'],                               'FTD': ['A9']}
cl_map_v3['QA']             = {'CORE': ['FI','QA','FH'],                                               'UDM': ['QA'],                                  'FTD': ['A20']}
cl_map_v3['news']           = {'CORE': ['NE'],                                                         'UDM': ['news'],                                'FTD': ['A8']}

# Convenience lookup by name
maps = {"cl_map_v1": cl_map_v1, "cl_map_v2": cl_map_v2, "cl_map_v3": cl_map_v3}

# ---------------------------------------------------------------------------
# Language family groupings
# ---------------------------------------------------------------------------
language_families = {
    'Afro-Asiatic':  ['Hebrew', 'Maltese'],
    'Altaic':        ['Turkish', 'Uyghur'],
    'Austronesian':  ['Indonesian'],
    'IE.Celtic':     ['Scottish Gaelic'],
    'Code-switch':   ['Hindi English', 'Turkish German'],
    'Creole':        ['Naija'],
    'IE.Baltic':     ['Lithuanian'],
    'IE.Germanic':   ['Afrikaans', 'Dutch', 'English', 'German', 'Icelandic', 'Norwegian', 'Swedish'],
    'IE.Greek':      ['Greek'],
    'IE.Armenian':   ['Armenian', 'Western Armenian'],
    'IE.Indic':      ['Hindi'],
    'IE.Romance':    ['Catalan', 'French', 'Italian', 'Portuguese', 'Romanian', 'Spanish'],
    'IE.Slavic':     ['Belarusian', 'Bulgarian', 'Croatian', 'Czech', 'Polish', 'Russian', 'Slovak', 'Slovenian'],
    'Sino-Tibetan':  ['Chinese'],
    'Uralic':        ['Estonian', 'Finnish', 'Erzya'],
    'Dravidian':     ['Tamil'],
}
