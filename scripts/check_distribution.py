"""Check that release archives contain only software and synthetic examples."""
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {'README.md', 'LICENSE', 'NOTICE', 'CITATION.cff', 'MANIFEST.in',
           'pyproject.toml', 'setup.cfg', 'PKG-INFO', 'src', 'tests', 'examples',
           'docs', 'scripts'}
FORBIDDEN_SUFFIXES = {'.tex', '.bib', '.pdf', '.docx', '.pptx', '.xlsx', '.mp4'}

def check(names, sdist=False):
    for name in names:
        parts = Path(name).parts[1:] if sdist else Path(name).parts
        if not parts:
            continue
        assert Path(name).suffix.lower() not in FORBIDDEN_SUFFIXES, name
        assert not {'paper', 'data', 'results', 'configs'}.intersection(parts), name
        if sdist:
            assert parts[0] in ALLOWED, name
        else:
            assert parts[0] == 'phasegraph' or parts[0].endswith('.dist-info'), name

for path in (ROOT/'dist').iterdir():
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            check(archive.namelist())
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path) as archive:
            check(archive.getnames(), sdist=True)
    elif path.name != 'SHA256SUMS.txt':
        raise AssertionError(f'Unexpected release file: {path.name}')
print('Distributions contain only the tool, documentation and synthetic examples.')
