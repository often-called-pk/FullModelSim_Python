"""Application entry. Normal launch -> GUI; `--headless cfg.json` -> solve."""
import sys


def is_headless(argv):
    return "--headless" in argv


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if is_headless(argv):
        i = argv.index("--headless")
        if i + 1 >= len(argv):
            print("error: --headless requires a config file path", file=sys.stderr)
            return 2
        cfg_path = argv[i + 1]
        import headless_solve
        return headless_solve.main([cfg_path])
    from PySide6.QtWidgets import QApplication
    from app.mainwindow import MainWindow
    app = QApplication(argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
