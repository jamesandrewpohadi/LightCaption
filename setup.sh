coco="coco"
if [ -d "$coco" ]; then
  echo "${coco} exists, build coco..."
  cd coco/PythonAPI/
  make
  python3 setup.py build
  python3 setup.py install
  cd ../../
else
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI/
  make
  python3 setup.py build
  python3 setup.py install
  cd ../../
fi

resize="data/resized2014"
if [ -d "$resize" ]; then
  echo "${resize} exists"
else
    echo "resized images to ${resize}"
  cd ../
  python3 scratch/resize.py
  cd scratch
fi

vocab="data/vocab.pkl"
if [ -f "$vocab" ]; then
  echo "${vocab} exists"
else
  echo "building vocab..."
  cd ../
  python scratch/build_vocab.py 
  cd scratch
  echo "saved vocab to ${vocab}"
fi