def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m: return ''
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def str2time(s):
    try:
        d, time = s.split('d')
        if d:
            d = int(d)
        else:
            d = 0
        h, m, s = (int(_) for _ in time.split(':'))
        t = d * 86400 + h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError('Invalid time string: %s ' % s)

    return t


def time2str(t, pretty=False):
    def pad(integer):
        r = str(integer)
        return r if len(r) > 1 else '0' + r

    d = int(t // 86400)
    h = int((t // 3600) % 24)
    m = int(t % 3600 // 60)
    s = int(round(t % 3600 % 60))
    t = {'d': d, 'h': h, 'm': m, 's': s, 't': t}

    time = '%s:%s:%s' % (pad(t['h']), pad(t['m']), pad(t['s']))
    if pretty:
        if t['d'] == 1:
            return '1 day ' + time
        elif t['d'] > 1:
            return str(t['d']) + ' days ' + time
        else:
            return time
    else:
        if d > 0:
            return '%id' % t['d'] + time
        else:
            return time


def to_json(json_dict):
    import json
    json_str = json.dumps(json_dict, sort_keys=True, indent=4)
    segmented_json = []
    last_id = 0
    id = json_str.find('[')
    while id != -1:
        id += 1
        current_segment = json_str[last_id:id]
        while json_str[id] != ']':
            if json_str[id] == '\n' or (json_str[id] == ' ' and json_str[id - 1] == ' '):
                if current_segment:
                    segmented_json.append(current_segment)
                    current_segment = ''
            else:
                current_segment += json_str[id]
            id += 1

        segmented_json.append(current_segment)
        last_id = id
        id = json_str.find('[', id)

    segmented_json.append(json_str[last_id:])
    return ''.join(segmented_json)