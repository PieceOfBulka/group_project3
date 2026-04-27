import time
import gc
import random
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import fake_useragent
import os
import file_sripts
from logger_master import get_logger
import slice_data_from_html

log = get_logger('AGENT_COLLENT_DATA')



def build_dataframe(batch_list):
    """
    Нужные столбцы:
    hh_vac_id
    hh_vac_link
    title
    experience
    salary_raw
    employer
    address_raw
    """

    df = pd.DataFrame(batch_list)

    df = df.rename(columns={
        "company_name": "employer"
    })

    required_cols = [
        "hh_vac_id",
        "hh_vac_link",
        "title",
        "experience",
        "salary_raw",
        "employer",
        "address_raw"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df = df[required_cols]
    df = df.drop_duplicates(subset=["hh_vac_id"]).reset_index(drop=True)

    return df


def get_info_from_user(user_text='Аналитик данных'):
    job_title = list()
    job_title.append(user_text)
    
    print(f'--- Поиск последних вакансий по профессии: {job_title}')

    return job_title


def scroll_down(driver, steps=5, delay=0.5):
    for _ in range(steps):
        driver.execute_script("window.scrollBy(0, window.innerHeight * 0.7);")
        time.sleep(delay)


def create_driver():
    new_options = webdriver.ChromeOptions()
    user_agent = fake_useragent.UserAgent()

    new_options.add_argument(f'user-agent={user_agent.random}')
    new_options.add_argument('--disable-blink-features=AutomationControlled')
    new_options.add_experimental_option('excludeSwitches', ['enable-automation'])

    driver = webdriver.Chrome(options=new_options)

    if driver:
        log.info(
            f'Драйвер создан: {driver}'
        )
        return driver
    else:
        log.critical(
            'Драйвер НЕ создан'
        )


def prepare_user_response(text: str):

    try:
        if isinstance(text, str) and len(text) >= 1:

            text = text.strip()

            words = '+'.join(text.split())

            synonims = '&ored_clusters=true'

            result_link = '?text=' + words + synonims

            return result_link

        return -1

    except Exception as ex:
        log.error(ex)
        return -1


def get_info_job_title(total_link, driver, name_of_dir):

    all_html = ''
    page = 0

    try:

        while True:

            url = (
                f'https://hh.ru/search/vacancy'
                f'{total_link}'
                f'area=113'
                f'&label=with_salary'
                f'&ored_clusters=true'
                f'&search_field=name'
                f'&order_by=publication_time'
            )

            driver.get(url=url)

            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located(
                    (By.TAG_NAME, 'body')
                )
            )

            scroll_down(driver)

            result = driver.page_source
            all_html += result

            if page > 0 and page % 5 == 0:
                vac_name = total_link \
                    .replace('?text=', '') \
                    .replace('&ored_clusters=true', '') \
                    .replace('+', '_')

                path_to_load_check = os.path.join(
                    name_of_dir,
                    f'page_{vac_name}_checkpoint_{page}.html'
                )

                with open(
                    path_to_load_check,
                    'w',
                    encoding='utf-8'
                ) as file:
                    file.write(all_html)

                all_html = ''

                gc.collect()

            soup = BeautifulSoup(
                result,
                'html.parser'
            )

            future_page = soup.find(
                'a',
                attrs={'data-qa': 'pager-page'},
                string=str(page + 2)
            )

            del soup
            gc.collect()

            if future_page:
                page += 1
            else:
                break

            if page == 5:
                break

        if all_html:

            vac_name = total_link \
                .replace('?text=', '') \
                .replace('&ored_clusters=true', '') \
                .replace('+', '_')

            path_to_load_check = os.path.join(
                name_of_dir,
                f'page_{vac_name}_checkpoint_{page}.html'
            )

            with open(
                path_to_load_check,
                'w',
                encoding='utf-8'
            ) as file:
                file.write(all_html)

        return 1

    except Exception as ex:
        log.error(ex)



if __name__ == '__main__':
    name_of_dir = file_sripts.create_checkpoints_dir()

    try:
        driver = create_driver()
        job_name = get_info_from_user() # запрос от пользователя; по умолчанию 'Аналитик данных'

        for curr_vac_name in job_name:
            second_link_part = prepare_user_response(curr_vac_name)

            if second_link_part != -1:
                get_info_job_title(
                    second_link_part,
                    driver,
                    name_of_dir
                )

            time.sleep(random.uniform(5, 12))

    finally:
        driver.quit()

    log.info('Парсинг HTML -> DataFrame')

    html_files = sorted(
        f for f in os.listdir(name_of_dir) if f.endswith('.html')
    )

    if not html_files:
        log.error(f'НЕ НАЙДЕНЫ HTML ФАЙЛЫ В {name_of_dir}')
    else:
        seen_ids = set()
        batch_list = []

        for filename in html_files:
            path = os.path.join(name_of_dir, filename)
            log.info(f'Обработка файла: {path}')

            vac = slice_data_from_html.get_first_data_batch(path)

            for item in vac:
                if item['hh_vac_id'] not in seen_ids:
                    seen_ids.add(item['hh_vac_id'])
                    batch_list.append(item)

        df = build_dataframe(batch_list)

        print(df.head())
        print(df.shape)

        df.to_csv('vacancies.csv', index=False, encoding='utf-8')
        df.to_parquet('vacancies.parquet', index=False)

        log.info('DataFrame сохранён')