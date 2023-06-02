import os.path
import time

import click
import colorama
from colorama import Fore, Style

import tool_module
from deep_face_module_v2 import MODELS, DETECTOR_BACKEND, all_df_encodings_to_file
from web.dbManager import DBManager
from web.resultDBManager import ResultDBManager

PHOTO_DIRECTORY = 'web/static/out/photo'
DB_DIRECTORY = 'web/db'
METADATA = 'web/static/out/metadata.json'
RESULT_DB = 'web/NmBookPhoto-result.db'

colorama.just_fix_windows_console()


@click.group()
def cli():
    pass


@cli.command()
@click.argument('model', default=MODELS[8])
@click.option('--detector_backend', '-d_back', default=DETECTOR_BACKEND[0], help='Детектор определения лиц.')
def create_db(model, detector_backend):
    """ Создает базу данных для модели распознавания лиц MODEL с использованием DETECTOR_BACKEND """
    click.echo(f'{Fore.YELLOW}START\nPress CTRL+C to quit{Style.RESET_ALL}', color=True)
    start_time = time.time()

    # Проверка аргументов
    try:
        m = MODELS[int(model)]
        model = m
    except ValueError:
        if not (model in MODELS):
            raise ValueError(f'{model} - не существующая model. '
                             f'model должна быть из {MODELS} или из{range(len(MODELS))}')
    except IndexError:
        raise ValueError(f'{model} - не существующий индекс model. '
                         f'model должна быть из {MODELS} или из{range(len(MODELS))}')
    try:
        d_bac = DETECTOR_BACKEND[int(detector_backend)]
        detector_backend = d_bac
    except ValueError:
        if not (detector_backend in DETECTOR_BACKEND):
            raise ValueError(f'{detector_backend} - не существующий detector_backend. '
                             f'detector_backend должен быть из {DETECTOR_BACKEND} или из{range(len(DETECTOR_BACKEND))}')
    except IndexError:
        raise ValueError(f'{detector_backend} - не существующий индекс detector_backend. '
                         f'detector_backend должен быть из {DETECTOR_BACKEND} или из{range(len(DETECTOR_BACKEND))}')
    # ---------------------

    # Создание json-файла с кодировками для выбранной модели
    click.echo(f'{Fore.GREEN}Model: {model}', color=True)
    click.echo(f'Detector backend: {detector_backend}', color=True)
    click.echo(f'Создание json-файла с кодировками для выбранной модели: {Style.RESET_ALL}', color=True)

    try:
        encoding_file_name = all_df_encodings_to_file(tool_module.get_all_file_in_directory(PHOTO_DIRECTORY),
                                                      PHOTO_DIRECTORY + '/encodings.json',
                                                      model_name=model,
                                                      detector_backends=[detector_backend],
                                                      process_count=2)['file_name']
    except ValueError as ve:
        if str(ve).startswith('File') and str(ve).endswith('already exists'):
            encoding_file_name = str(ve).split()[1][1:-1]  # срез убирает кавычки
            click.echo(
                f'{Fore.CYAN}WARNING: json-файл {Fore.RED}{encoding_file_name}{Fore.CYAN} с кодировками для выбранной '
                f'модели уже существует. Далее будет использован старый файл с кодировками. Если были добавлены новые '
                f'фото, то удалите старый файл и создайте новый.{Style.RESET_ALL}',
                color=True)
        else:
            raise ve
    click.echo(
        f'{Fore.GREEN}Создание json-файла с кодировками для выбранной модели УСПЕШНО ЗАВЕРШЕНО!!!{Style.RESET_ALL}',
        color=True)
    # ---------------------

    # Создание базы данных для выбранной модели
    click.echo(f'{Fore.GREEN}Создание базы данных для выбранной модели: {Style.RESET_ALL}', color=True)
    if not (os.path.exists(DB_DIRECTORY)):
        os.mkdir(DB_DIRECTORY)
    db_name = f'{DB_DIRECTORY}/NmBookPhoto({model}-{detector_backend}).db'
    create_model_db = True
    if os.path.exists(db_name):
        click.echo(f'{Fore.CYAN}WARNING: база данных {Fore.RED}{db_name}{Fore.CYAN} уже существует.{Style.RESET_ALL}')
        if click.confirm('Удалить существующую базу данных и создать новую?', default=False):
            os.remove(db_name)
            click.echo(f'{Fore.GREEN}База данных {db_name} успешно удалена!!!{Style.RESET_ALL}')
        else:
            create_model_db = False
            click.echo(f'{Fore.GREEN}Существующая база данных будет использоваться!!!{Style.RESET_ALL}')
    if create_model_db:
        mdb_manager = DBManager(db_name)
        mdb_manager.fill_doc_photos_faces_from_file(encoding_file_name, METADATA)
        click.echo(f'{Fore.GREEN}Расчет матрицы расстояний:{Style.RESET_ALL}')
        mdb_manager.calculate_distance_matrix(process_count=10)
        click.echo(f'{Fore.GREEN}Создание и заполнение базы данных {db_name} УСПЕШНО ЗАВЕРШЕНО!!!{Style.RESET_ALL}',
                   color=True)
    # ---------------------

    # Создание базы данных результатов
    if not (os.path.exists(RESULT_DB)):
        click.echo(f'{Fore.GREEN}Создание базы данных результатов {RESULT_DB}: {Style.RESET_ALL}', color=True)
        ResultDBManager(RESULT_DB)
        click.echo(f'{Fore.GREEN}Создание базы данных результатов УСПЕШНО ЗАВЕРШЕНО!!!{Style.RESET_ALL}', color=True)

    end_time = time.time()
    click.echo(f'{Fore.YELLOW}Time: {end_time - start_time} s ({(end_time - start_time) / 60} min)', color=True)
    click.echo(f'Все операции успешно завершены!!!{Style.RESET_ALL}', color=True)


if __name__ == '__main__':
    cli()
