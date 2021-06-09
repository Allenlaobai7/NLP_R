import re

''' 如，等 '''
l1 = '如(Jira/Bugfree/)等,主流测试工具，如(robotium,UI Automation)等'    # 如+等  括号
l2 = '如robotium, UI Automation)'                                      # 如
l3 = '如：TD、loadrunner、QTP 等'                                # 如+等，非括号
l4 = '如(c++/python/Java)'                  # 如 c++
l5 = '如xUnit) 或常见的自动化测试工具'           # 如 括号
l6 = '如(Jira/Bugfree/)等,主流测试工具，如(robotium,UI Automation)等'
l7 = 'TD、loadrunner、QTP等'                   #等
l8 = '如java/python/c++/)及一门语言(ah/dfsi/sfd dsfe等)'   #最难的情况
l9 = '熟练使用（wirShark、fiddler、soapUI、c++）等网络工具'
l10 = '深刻掌握推荐技术、机器学习、等中的至少一项技术'


def split_sentence(text):
    newtext = []
    if not re.search(r'等',text):            #只有如
        newtext = re.findall(r'如[^如]+',text)
    elif not re.search(r'如',text):          #只有如
        newtext = re.findall(r'[^等]+等', text)
    else:                                    #有如和等
        subtext = re.findall(r'(如[^如^等]+等)|(如[^如^等]+[^如])|(如[^如^等]+)|([^如^等]+等)',text)
        for x in subtext:
            for i in x:
                if i is not '':
                    newtext.append(i)
    return newtext

def search_skill(text):
    skills = []
    text = re.sub(",|/|\\|、|&|或", '、',text)
    separator = '、'
    p = r'' + re.escape(separator)

    # 有如和等
    if re.findall(r'(?<=如)[^如]+(?=等)', text):
        text = re.sub('\（','',text)
        text = re.sub('\(', '', text)
        text = re.sub('\）', '', text)
        text = re.sub('\)', '', text)

        raw_left = re.split(p, text)[0]
        middle = re.split(p, text)[1:-1]
        raw_right = re.split(p, text)[-1]

        if not re.search(p, text):
              skills = re.search(r'如\W*(?P<left>\w+\S*[^)])等',raw_left).group("left")
        else:
            left = re.search(r'如\W*(?P<left>\w+.+)\s*', raw_left).group("left")
            skills.extend([left])
            skills.extend(middle)
            right_match = re.search(r'(?P<right>\S+)\s*等', raw_right)
            if right_match:
                right = right_match.group("right")
                skills.extend([right])


    # 只有 如
    elif not re.search(r'等', text):
        if re.search(r'如.+[)）]',text):
            text = re.split("\)|\）",text)[0]
            text = re.sub('\（', '', text)
            text = re.sub('\(', '', text)

        raw_left = re.split(p, text)[0]
        middle = re.split(p, text)[1:-1]
        raw_right = re.split(p, text)[-1]
        if re.search(p,text):
            left_match = re.search(r'如\W*(?P<left>\w+\S+)', raw_left)
            if left_match:
                left = left_match.group('left')
                skills.extend([left])
            skills = skills + middle
            p_right = r'(?P<right>\w+\s*\w*\s*\W*)'
            right_match = re.search(p_right, raw_right)
            if right_match:
                 right = right_match.group("right")
                 skills.extend([right])
        else:
            left_match = re.search(r'如\s*\W*\s*(?P<left>\w+\s*\w*\s*\W*)', raw_left)
            if left_match:
                 left = left_match.group('left')
                 skills.extend([left])

    # 只有 等
    else:
        if re.search(r'[\(\（].+等',text):
            text = re.split("\(|\（",text)[-1]
            text = re.sub('\）', '', text)
            text = re.sub('\)', '', text)

        raw_left = re.split(p, text)[0]
        middle = re.split(p, text)[1:-1]
        raw_right = re.split(p, text)[-1]
        if re.search(p, text):
            right_match = re.search(r'(?P<right>\S+)\s*等', raw_right)
            if right_match:
                right = right_match.group("right")
                skills.extend([right])
            skills = skills + middle
            p_left = r'(?P<left>\w+\s*\w*\s*\W*)' + r'\s*' + re.escape(separator)
            left_match = re.search(p_left, text)
            if left_match:
                left = left_match.group('left')
                skills.extend([left])
            skills = skills[::-1]
        else:
            print(raw_right)
            right_match = re.search(r'(?P<right>\w+\s*\w*\s*\W*)\s*等', raw_right)
            if right_match:
                right = right_match.group("right")
                skills.extend([right])

    # print(skills)
    return skills



def listed_ops(sentences):
    result = split_sentence(sentences)
    skills_more = []
    for text in result:
        skills_more.extend(search_skill(text))
    print(skills_more)
    return skills_more


p0 = re.compile(r'如|等')
text = l7
if p0.search(text):
    listed_ops(text)





