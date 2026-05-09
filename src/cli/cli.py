"""
cli.py
======
Entry point da ferramenta transmut.

Comandos disponíveis:
  transmut run    — executa o pipeline de mutação
  transmut init   — cria transmut.toml guiado
  transmut show   — abre o último report.html no browser
"""
from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transmut",
        description="Mutation testing para pipelines PySpark com DataFrame API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemplos:
  transmut init                              # cria transmut.toml interativo
  transmut run                               # usa transmut.toml na raiz
  transmut run --src etl_code/ --tests tests/
  transmut run --src etl_code/transforms.py --tests tests/test_transforms.py
  transmut run --config config.txt           # modo legado
  transmut run --operators NFTP MTR          # só dois operadores
  transmut show                              # abre o último relatório
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── transmut run ──────────────────────────────────────────────────
    run = sub.add_parser("run", help="Executa o pipeline de mutação")

    src_group = run.add_argument_group("fontes (escolha uma forma)")
    src_group.add_argument(
        "--src", metavar="PATH",
        help="Arquivo .py ou diretório com o código PySpark",
    )
    src_group.add_argument(
        "--tests", metavar="PATH",
        help="Arquivo .py ou diretório com os testes pytest",
    )
    src_group.add_argument(
        "--config", metavar="FILE",
        help="Caminho para transmut.toml ou config.txt",
    )

    run.add_argument(
        "--operators", metavar="OP", nargs="+",
        default=["MTR", "NFTP", "ATR", "UTS"],
        help="Operadores a usar (default: MTR NFTP ATR UTS)",
    )
    run.add_argument(
        "--output", metavar="DIR", default=".",
        help="Diretório raiz onde TransmutPysparkOutput será criado (default: .)",
    )
    run.add_argument(
        "--workers", type=int, default=4,
        help="Workers paralelos para execução dos testes (default: 4)",
    )
    run.add_argument(
        "--verbose", "-v", action="store_true",
        help="Ativa logs detalhados",
    )

    # ── transmut init ─────────────────────────────────────────────────
    init = sub.add_parser("init", help="Cria transmut.toml com configuração padrão")
    init.add_argument("--src",   metavar="PATH", default="src/",
                      help="Diretório do código fonte (default: src/)")
    init.add_argument("--tests", metavar="PATH", default="tests/",
                      help="Diretório dos testes (default: tests/)")
    init.add_argument("--output", metavar="DIR", default=".",
                      help="Diretório de saída (default: .)")

    # ── transmut show ─────────────────────────────────────────────────
    sub.add_parser("show", help="Abre o último report.html no browser")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    _setup_logging(verbose=getattr(args, "verbose", False))

    if args.command == "init":
        _cmd_init(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "show":
        _cmd_show()


# ─────────────────────────────────────────────────────────────────────── #
# Comandos                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def _cmd_run(args: argparse.Namespace) -> None:
    """Resolve a fonte de config e dispara o MutationManager."""
    from src.mutation_manager import MutationManager

    config_input = _resolve_config_input(args)

    try:
        _print_banner()
        manager = MutationManager(config_input)
        manager.run()

        # Resumo no terminal
        total    = len(manager.mutant_list)
        killed   = sum(1 for r in manager.result_list if r.status == "killed")
        score    = round(killed / total * 100, 1) if total else 0
        survived = total - killed

        print(f"\n{'━'*50}")
        print(f"  Mutation Score : {score}%  ({killed} killed / {total} total)")
        print(f"  Sobreviventes  : {survived} mutante(s) não detectados")
        print(f"  Relatório      : {manager.work_dir}/report.html")
        print(f"{'━'*50}\n")

        if score < 60:
            print("  ⚠ Score baixo — confira os mutantes sobreviventes no relatório.\n")

    except FileNotFoundError as e:
        _die(f"Arquivo não encontrado: {e}")
    except ValueError as e:
        _die(f"Configuração inválida: {e}")
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(1)


def _cmd_init(args: argparse.Namespace) -> None:
    """Cria transmut.toml no diretório atual."""
    toml_path = Path("transmut.toml")
    if toml_path.exists():
        print("transmut.toml já existe. Remova-o antes de rodar 'init'.")
        sys.exit(1)

    content = (
        "[transmut]\n"
        f'source_dirs   = ["{args.src}"]\n'
        f'tests_dirs    = ["{args.tests}"]\n'
        'operators     = ["MTR", "NFTP", "ATR", "UTS"]\n'
        f'workspace_dir = "{args.output}"\n'
        '# exclude = ["src/utils/logging.py"]   # opcional\n'
    )
    toml_path.write_text(content, encoding="utf-8")
    print("✓ transmut.toml criado.")
    print("  Edite os caminhos e rode 'transmut run'.")


def _cmd_show() -> None:
    """Abre o último relatório no browser."""
    # Procura qualquer report*.html dentro de TransmutPysparkOutput
    candidates = [
        p for p in sorted(Path(".").rglob("*.html"))
        if "TransmutPysparkOutput" in str(p) and p.stem.startswith("report")
    ]

    if not candidates:
        _die(
            "Nenhum relatório encontrado.\n"
            "  Certifique-se de estar na raiz do projeto onde 'transmut run' foi executado.\n"
            "  Rode 'transmut run' primeiro para gerar o relatório."
        )

    # Pega o mais recente (por nome — o timestamp garante ordenação correta)
    report = sorted(candidates)[-1].resolve()
    print(f"Abrindo: {report}")
    webbrowser.open(report.as_uri())


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                  #
# ─────────────────────────────────────────────────────────────────────── #

def _resolve_config_input(args: argparse.Namespace) -> str | dict:
    """
    Determina a fonte de configuração na ordem de prioridade:
      1. --src + --tests (flags diretas)
      2. --config explícito
      3. transmut.toml na raiz
      4. config.txt na raiz
    """
    # Modo A: flags diretas
    if args.src and args.tests:
        return {
            "program_path":  args.src,
            "tests_path":    args.tests,
            "operators_list": ",".join(args.operators),
            "workspace_dir": args.output,
        }

    # Modo B/C: arquivo de config explícito
    if args.config:
        p = Path(args.config)
        if not p.exists():
            _die(f"Arquivo de config não encontrado: {args.config}")
        return str(p)

    # Modo B: transmut.toml automático
    if Path("transmut.toml").exists():
        return "transmut.toml"

    # Modo C: config.txt legado
    if Path("config.txt").exists():
        return "config.txt"

    _die(
        "Nenhuma configuração encontrada.\n"
        "  Use --src e --tests, --config, ou crie transmut.toml com 'transmut init'."
    )


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    # Sempre mostra INFO do MutationManager (progresso do pipeline)
    logging.getLogger("src.mutation_manager").setLevel(logging.INFO)
    logging.getLogger("src.config.ast_analyzer").setLevel(logging.INFO)
    logging.getLogger("src.test_module.test_runner").setLevel(logging.INFO)


def _print_banner() -> None:
    print("\nTransmutPySpark — Mutation Testing para pipelines PySpark")
    print("─" * 50)


def _die(msg: str) -> None:
    print(f"\nErro: {msg}\n", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
