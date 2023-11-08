import re

regex = '([^가-힣0-9a-zA-Z.,·•%↓()\s\\\])'


def _result_to_regex(result: list) -> list:
    storage = []
    for s in result:
        sub_str = re.sub(pattern=regex, repl="****", string=s)
        storage.append(sub_str)
    return storage


test_str = 'LPG용기 사용가구 시설개선\n\n대상\n\nLPG용기 사용 주택에서 LPG 고무호스를 사용 중인 가구\n\n내용\n\nLP가스 고무호스 교체(금속배관) 및 안전장치(퓨즈콕) 등 가스시설 ' \
           '설치 지원 (시공비 약 25만 원 중 20만 원 상당 지원, 자부담 5만 원)\n\n방법\n\n수혜대상자가 소재지 시군구 가스담당부서 또는 읍면동 주민센터(행정복지센터) 방문 ' \
           '신청\n\n문의\n\n소재지 시군구 가스담당부서 또는 읍면동 주민센터(행정복지센터)\n\n한국가스안전공사(☎1544-4500) '

test_list = test_str.split()

print(test_str)
print('-'*30)
re_str = " ".join(_result_to_regex(test_list))
print(re_str)
print('-'*30)
x = [re.sub(pattern=regex, repl="****", string=w) for w in test_list]
print(" ".join(x))
print('-'*30)
x = map(lambda xx: re.sub(regex, "****", xx), test_list)
print(" ".join(x))
