"""make_surfaceを実行する際のパラメータ."""


class Param:
    """
    初期化用
    """
    edit_normal = False
    work_process = False
    show_normal = False
    show_mesh = False

    """
    以下切り替え用
    """
    # 法線ベクトルの編集処理の有無
    edit_normal = True

    # 作業過程のグラフの表示
    work_process = True

    # 法線ベクトルの表示
    # show_normal = True

    # meshの表示
    # show_mesh = True
