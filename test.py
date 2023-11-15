from workspace.mdLoader import BaseDBLoader
import re



test_loader = BaseDBLoader()

docs = test_loader.load()

print(len(docs))