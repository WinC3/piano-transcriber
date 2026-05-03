# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

# Collect necessary data files
torch_data = collect_data_files('torch')

a = Analysis(
    ['piano_transcriber/gui/app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('checkpoints/model_epoch_1702.pth', 'checkpoints'),  # Include latest model
    ] + torch_data,
    hiddenimports=[
        'torch._C._distributed_c10d',
        'torch._C',
        'torchaudio',
        'pretty_midi',
        'librosa',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude only libraries we've verified are NOT imported in the codebase
        'sklearn',
        'jupyter',
        'ipython',
        'notebook',
        'tensorboard',
        'cv2',
        # Exclude test modules
        'test',
        'tests',
        'testing',
        '_pytest',
        'pytest',
        # Exclude documentation
        'sphinx',
        'docutils',
        # Keep setuptools and distutils as they might be needed
        # Keep scipy, pandas, matplotlib, PIL as they are used by dependencies
    ],
    noarchive=False,
    optimize=1,  # Enable bytecode optimization
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='piano-transcriber-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Strip debug symbols
    upx=True,    # Enable UPX compression if available
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='piano_transcriber/gui/icon.ico' if os.path.exists('piano_transcriber/gui/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,  # Strip debug symbols from binaries
    upx=True,    # Enable UPX compression if available
    upx_exclude=[],
    name='piano-transcriber-gui',
)
