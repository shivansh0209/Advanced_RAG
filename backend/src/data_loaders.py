import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

def load_documents(path="data", glob="**/*.md"):
    loader = DirectoryLoader(
        path=path,
        glob=glob,
        recursive=True,
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()