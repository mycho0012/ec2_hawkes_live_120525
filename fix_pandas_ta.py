#!/usr/bin/env python3
"""
Linux/Ubuntu 서버용 pandas_ta 패치 스크립트
"""
import os
import sys

def fix_pandas_ta():
    """pandas_ta 패치 적용"""
    # site-packages 경로 검색
    site_packages_paths = []
    for path in sys.path:
        if 'site-packages' in path or 'dist-packages' in path:
            site_packages_paths.append(path)
    
    # 홈 디렉토리의 로컬 Python 경로도 검색
    home = os.path.expanduser('~')
    for py_ver in ['3.8', '3.9', '3.10', '3.11']:
        local_path = os.path.join(home, '.local', 'lib', f'python{py_ver}', 'site-packages')
        if os.path.exists(local_path):
            site_packages_paths.append(local_path)
    
    # 모든 경로에서 squeeze_pro.py 파일 찾기
    fixed = False
    for site_pkg in site_packages_paths:
        squeeze_path = os.path.join(site_pkg, 'pandas_ta', 'momentum', 'squeeze_pro.py')
        if os.path.exists(squeeze_path):
            print(f"파일 발견: {squeeze_path}")
            
            # 파일 내용 수정
            with open(squeeze_path, 'r') as f:
                content = f.read()
            
            if 'from numpy import NaN as npNaN' in content:
                print("NaN 임포트 찾음, 수정 중...")
                fixed_content = content.replace('from numpy import NaN as npNaN', 'from numpy import nan as npNaN')
                
                with open(squeeze_path, 'w') as f:
                    f.write(fixed_content)
                
                print(f"파일 수정 완료: {squeeze_path}")
                fixed = True
    
    if fixed:
        print("패치 적용 성공! 이제 ec2_hawkes_live.py를 실행할 수 있습니다.")
        return True
    else:
        print("수정할 파일을 찾지 못했습니다.")
        return False

if __name__ == "__main__":
    fix_pandas_ta()