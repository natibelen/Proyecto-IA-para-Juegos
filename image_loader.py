import pygame

def convert_img(img, w, h):
    return pygame.transform.smoothscale(pygame.image.load(img).convert(), (w, h))

def convert_alpha_img(img, w, h):
    return pygame.transform.smoothscale(pygame.image.load(img).convert_alpha(), (w, h))

class ImageLoader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

            self.background = []
            for i in range(2):
                self.background.append(convert_img("resources/background_"+str(i)+".png", 1280, 720))

            self.arrow = []
            for i in range(2):
                self.arrow.append(convert_alpha_img("resources/arrow_"+str(i)+".png", 90, 90))

            self.cover = []
            for i in range(4):
                self.cover.append(convert_img("resources/album_cover_"+str(i)+".png", 280, 280))


    def get_background(self, index):
        return self.background[index]

    def get_arrow(self, index):
        return self.arrow[index]

    def get_cover(self, index):
        return self.cover[index]