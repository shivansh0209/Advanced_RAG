import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_documents(path="data", glob="**/*.pdf", loader_cls=PyPDFLoader):
    loader = DirectoryLoader(
        path=path,
        glob=glob,
        recursive=True,
        show_progress=True,
        use_multithreading=True,
        loader_cls=loader_cls,
        loader_kwargs={"strategy":"ocr_only", "language":"eng"} if loader_cls != PyPDFLoader else {}
    )
    return loader.load()