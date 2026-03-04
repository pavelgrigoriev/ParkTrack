class Colors:
    FILL_FREE    = (80, 220, 80)
    FILL_BUSY    = (60, 60, 230)
    BORDER_FREE  = (50, 255, 50)
    BORDER_BUSY  = (30, 30, 255)
    WHITE        = (255, 255, 255)
    BLACK        = (0, 0, 0)
    CYAN         = (255, 255, 0)
    YELLOW       = (0, 255, 255)
    PANEL_BG     = (40, 40, 40)
    GREEN_TEXT   = (80, 230, 80)
    RED_TEXT     = (80, 80, 240)
    GRAY_TEXT    = (180, 180, 180)
    ORANGE_TEXT  = (0, 165, 255)

    @classmethod
    def fill(cls, busy):
        return cls.FILL_BUSY if busy else cls.FILL_FREE

    @classmethod
    def border(cls, busy):
        return cls.BORDER_BUSY if busy else cls.BORDER_FREE
