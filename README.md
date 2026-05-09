# SilkRAD

SilkRAD — это компилятор lightmap для Source BSP с CUDA/OptiX трассировкой.

Текущее состояние проекта:
- основной backend один: `core`
- старый split `legacy/v2` убран
- дальнейшая работа должна идти через сверку с исходным `VRAD`, а не через сохранение старого поведения проекта

Куда смотреть в первую очередь:
- [docs/01-overview.md](docs/01-overview.md) — краткая карта проекта
- [docs/02-core-architecture.md](docs/02-core-architecture.md) — устройство `core`
- [docs/03-development-workflow.md](docs/03-development-workflow.md) — как вносить изменения без хаоса
- [docs/04-current-gaps.md](docs/04-current-gaps.md) — что ещё не доведено до `VRAD 1:1`
- [docs/05-vrad-mapping.md](docs/05-vrad-mapping.md) — карта соответствий `core` и исходного `VRAD`

Быстрый запуск:
```text
SilkRAD.exe <input.bsp> [output.bsp] -game <mod_root_or_gameinfo.gi>
```

Пример:
```text
SilkRAD.exe D:\games\CSS_LOVE\cstrike\maps\de_brigia_hvh_final.bsp D:\games\CSS_LOVE\cstrike\maps\out.bsp -game D:\games\CSS_LOVE\cstrike\gameinfo.gi
```

Главный принцип разработки:
- если меняется геометрия, sampling, tracing или ambient/direct logic, ориентиром должен быть исходный `VRAD`
- CUDA и OptiX — это только backend исполнения, а не источник алгоритмики
