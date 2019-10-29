  "import shutil
    "from random import randint
    "count = 0 
    "count = set()
    "pred_true = list(pred_true)
    "while len(count) < 5:
    "    random = randint(0,len(pred_true))
        count.add(random)
    
    count = list(count)
    for i in range(len(count)):
        shutil.copytree(lime_dir+\"/\"+pred_true[count[i]]+\"/\", \"./true/\"+pred_true[count[i]])
    
    count = 0 
    count = set()
    pred_false = list(pred_false)
    for i in range(len(pred_false)):
        #random = randint(0,len(pred_false))
        #count.add(random)
        shutil.copytree(lime_dir+\"/\"+pred_false[i]+\"/\", \"./false/\"+pred_false[i])"
