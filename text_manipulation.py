class MovingText:
    def __init__(self, text, max):
        self.start = 0
        self.end = max
        self.time = 0
        self.text = text
        self.move_for = True
        self.stop = False

    def move(self, moving, stopping):
        self.time += 1
        if self.time % moving == 0 and not self.stop:
            if self.move_for:
                self.start += 1
                self.end += 1
            else:
                self.start -= 1
                self.end -= 1

            if self.end == len(self.text):
                self.move_for = False
                self.stop = True
            elif self.start == 0:
                self.move_for = True
                self.stop = True

        elif self.time % stopping == 0 and self.stop:
            self.stop = False

        return self.text[self.start:self.end]
