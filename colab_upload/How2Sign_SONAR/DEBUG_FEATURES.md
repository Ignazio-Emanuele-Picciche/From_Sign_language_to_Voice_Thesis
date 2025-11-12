# 🔍 Debug: Trova le Feature Estratte

Copia questa cella su Colab per trovare dove sono le feature:

```python
# Verifica struttura directory
import os
from pathlib import Path

print("=" * 60)
print("📂 STRUTTURA DIRECTORY CORRENTE")
print("=" * 60)

# Directory corrente
print(f"\n📍 Working directory: {os.getcwd()}\n")

# Lista tutti i file e cartelle
print("📁 Contenuto directory corrente:")
!ls -lh

print("\n" + "=" * 60)
print("🔍 RICERCA FEATURE FILES")
print("=" * 60)

# Cerca ricorsivamente file .pt
import subprocess
result = subprocess.run(
    ['find', '.', '-name', '*.pt', '-type', 'f'],
    capture_output=True,
    text=True
)

pt_files = result.stdout.strip().split('\n')
pt_files = [f for f in pt_files if f]

if pt_files:
    print(f"\n✅ Trovati {len(pt_files)} file .pt:")
    for f in pt_files[:10]:  # Mostra primi 10
        print(f"   {f}")
    if len(pt_files) > 10:
        print(f"   ... e altri {len(pt_files) - 10} file")
else:
    print("\n❌ NESSUN file .pt trovato!")
    print("\n📋 Possibili problemi:")
    print("   1. Le feature non sono state estratte")
    print("   2. Le feature sono in un'altra cartella")
    print("   3. Le feature sono nel Mac, non su Google Drive")

print("\n" + "=" * 60)
print("📂 VERIFICA CARTELLE")
print("=" * 60)

folders_to_check = [
    'features',
    'features/train',
    'features/val',
    'features/test',
    'manifests',
    'models'
]

for folder in folders_to_check:
    exists = os.path.exists(folder)
    if exists:
        count = len(list(Path(folder).glob('*'))) if Path(folder).is_dir() else 0
        print(f"✅ {folder:30s} - {count} files")
    else:
        print(f"❌ {folder:30s} - NON ESISTE")
```

---

## 🎯 Esegui questa cella e dimmi cosa vedi!

Questo ti dirà:

1. ✅ Se le feature esistono e dove sono
2. ✅ Se la struttura directory è corretta
3. ✅ Quanti file `.pt` ci sono

Poi possiamo decidere il prossimo passo! 🔍
