import os
import io
import json

def get_files(path):
  for (dirpath, _, filenames) in os.walk(path):
    for filename in filenames:
      yield os.path.join(dirpath, filename)

def getMissingBooks():
  files = []
  for filename in get_files('./bookz/'):
    if '.txt' in filename:
      files += [filename]
  return files

def saveData(nFile, data):
  target = open(nFile, 'w')
  target.write(data)
  target.close()
  return target

def getBookName(book):
  return os.path.basename(book)

for book in getMissingBooks():
  with io.open(book, 'r', encoding='utf-8') as f:
    text = f.read()
  title = getBookName(book)
  auth = title.split(",")[0].strip()
  bookTitle = title.split(",")[1].replace('.txt','').strip()
  filename = "./books/" + bookTitle + "_" + auth + ".txt"
  print(title, auth, bookTitle, filename)
  saveData(filename,json.dumps(text).encode('utf-8'))

