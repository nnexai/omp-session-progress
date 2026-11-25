"""Export the locked posterior head; no rows, IDs, text or path dictionaries."""
import json,sys
from pathlib import Path
from utility import BASE
root=Path(sys.argv[1]);policy=json.loads((root/'posterior-selection.json').read_text())['best']['p'];assert policy==dict(head=28,weight=1,onset=450,useForecast=False)
m=json.loads((root/'survival3.json').read_text())[3];d=len(m['mu']);m['base']=BASE;m['beta']=m['params'][:d];m['skew']=m['params'][d:2*d];m['intercept']=m.pop('params')[2*d:];print(json.dumps({k:[round(v,9) for v in a] for k,a in m.items()},separators=(',',':')))
