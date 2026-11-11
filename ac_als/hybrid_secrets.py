"""Current-user Windows DPAPI storage; encrypted credential stays on D:."""
import ctypes,os
from ctypes import wintypes
from pathlib import Path
PATH=Path(os.environ.get('NETLIST_FRONTIER_ROOT',str(Path(__file__).resolve().parent)))/'.local/siliconflow.dpapi'

class Blob(ctypes.Structure):_fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]

def transform(raw,protect):
 if os.name!='nt':raise RuntimeError('DPAPI is available only on the current Windows host')
 data=(ctypes.c_ubyte*len(raw)).from_buffer_copy(raw);source=Blob(len(raw),data);target=Blob()
 if protect:ok=ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source),None,None,None,None,1,ctypes.byref(target))
 else:ok=ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source),None,None,None,None,1,ctypes.byref(target))
 if not ok:raise RuntimeError('Windows credential encryption/decryption failed')
 try:return ctypes.string_at(target.data,target.size)
 finally:
  ctypes.memset(target.data,0,target.size);ctypes.windll.kernel32.LocalFree(target.data)

def store_key(secret):
 if not secret or '\n' in secret or '\r' in secret:raise ValueError('Invalid credential')
 PATH.parent.mkdir(exist_ok=True);PATH.write_bytes(transform(secret.encode(),True))

def load_key():
 value=os.environ.get('SILICONFLOW_API_KEY','')
 if not value and PATH.exists():value=transform(PATH.read_bytes(),False).decode()
 if not value or '\n' in value or '\r' in value:raise RuntimeError('No valid SiliconFlow credential; no model request or fallback search started')
 return value
