def tokenizer(document: str) -> list[str]:
    i = 0
    temp_str = ""
    res = []
    document = document.lower()
    
    while i < len(document):
        if document[i] == ' ':
            if temp_str:
                res.append(temp_str)
                temp_str = ''
            i += 1
            continue
        elif document[i] < 'a' or document[i] > 'z':
            i += 1
            continue   
        else:
            temp_str += document[i]
            i += 1
    if temp_str:
        res.append(temp_str)

    return res
