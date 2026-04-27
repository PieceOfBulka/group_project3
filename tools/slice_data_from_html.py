from bs4 import BeautifulSoup
import sqlite3
import os
import db_sripts
from logger_master import get_logger

log = get_logger('SLICE')


def get_title(data_item):
    title_tag = data_item.find(attrs={'data-qa': 'serp-item__title-text'})
    if title_tag:
        title = title_tag.get_text(strip=True)
    else:
        title = ''
        log.debug(f"Название вакансии не найдено в {data_item}")
    
    return title


def get_vac_id(data_item):
    link_tag = data_item.find('a', attrs={'data-qa': 'serp-item__title'})
    if link_tag:
        href = link_tag['href']
    else:
        href = ''
        log.debug(f"Ссылка не найдена в {data_item}")
    
    vac_id = data_item.find('div', id=True)
    if vac_id:
        vac_id = vac_id['id']
        link = f'https://hh.ru/vacancy/{vac_id}'
    else:
        link = href.split("?")[0]
        vac_id = link.split('/')[-1]

        log.debug(f"ID-вакансии не найдено в {data_item}")
    
    return vac_id, link


def get_company_name(data_item):
    company = data_item.find(attrs={'data-qa': 'vacancy-serp__vacancy-employer-text'})
    if company:
        name = company.get_text(strip=True)
    else:
        name = ''
        log.debug(f"Название компании не найдена в {data_item}")
    
    return name


def get_salary(data_item):
    CURRENCY_SUMBOLS = [
        '₽',
        '$',
        '€',
        'руб',
        'RUB',
        'USD',
        'EUR',
        'KZT'
    ]
    salary = ''

    for span_tag in data_item.find_all('span'):
        text = span_tag.get_text(strip=True)
        if any(symb in text for symb in CURRENCY_SUMBOLS):
            salary = text
            break
    if salary == '':
        log.debug(f"Зарплата не найдена в {data_item}")
    
    return salary


def get_exp(data_item):
    exp = data_item.find(attrs={"data-qa": lambda x: x and "work-experience" in x})
    if exp:
        exp_data = exp.get_text(strip=True)
    else:
        exp_data = ''
        log.debug(f"Опыт не найдена в {data_item}")
    
    return exp_data


def get_company_adress(data_item):
    address_tag = data_item.find(attrs={"data-qa": "vacancy-serp__vacancy-address"})
    if address_tag:
        address = address_tag.get_text(strip=True)
    else:
        address = ''
        log.debug(f"Адрес не найден в {data_item}")

    return address


def get_first_data_batch(path_to_file):
    with open(path_to_file, encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    data_cards = soup.find_all('div', attrs={'data-qa': 'vacancy-serp__vacancy'})
    log.info(f'--- Найдено {len(data_cards)} вакансий')

    if len(data_cards) == 0:
        log.warning(f'НАЙДЕНО 0 карточек в {path_to_file}')

    result = []

    for item in data_cards:
        title = get_title(item)
        vac_id, url = get_vac_id(item)
        company_name = get_company_name(item)
        salary = get_salary(item)
        experience = get_exp(item)
        address = get_company_adress(item)

        result.append({
            'title': title,
            'hh_vac_id': vac_id,
            'hh_vac_link': url,
            'company_name': company_name,
            'salary_raw': salary,
            'experience': experience,
            'address_raw': address,
            })
    
    return result


def parse_all_checkpoints(cursor, conn, checkpoints_dir='checkpoints'):
    files = [item_file for item_file in os.listdir(checkpoints_dir) if item_file.endswith('.html')]
    files.sort()

    if len(files) == 0:
        log.error(f'--- ОШИБКА: НЕ НАЙДЕНЫ HTML ФАЙЛЫ В {checkpoints_dir}')
        return

    already_get_id = set()
    counter = 0
    all_size = len(files)

    for filename in files:
        path = os.path.join(checkpoints_dir, filename)
        log.info(f'! ОБРАБОТКА ФАЙЛА {path} --- {counter / all_size * 100}% !')

        vac = get_first_data_batch(path)

        unique_vac = [item for item in vac if item['hh_vac_id'] not in already_get_id]
        already_get_id.update(item['hh_vac_id'] for item in unique_vac)

        db_sripts.insert_data_first_batch(cursor, conn, unique_vac)
        
        counter += 1

    
# if __name__ == '__main__':
#     connection = sqlite3.Connection('GP_DB.db')
#     cursor = connection.cursor()

#     parse_all_checkpoints(cursor, connection)
    
#     print(db_sripts.select_limit_data(cursor))

#     connection.close()