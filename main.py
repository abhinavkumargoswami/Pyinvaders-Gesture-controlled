import pygame
import random
import cv2
from HandTrackingModule import HandDetector
import time
import math

detector = HandDetector(max_num_hands=1,
                        min_detection_confidence = 0.7)

# Initialize pygame
pygame.init()

width = 400
height = 300
# create screen
screen = pygame.display.set_mode((width, height))

# title and icon
pygame.display.set_caption("data/PyInvaders")
# Icon made by Freepik from www.flaticon.com
icon = pygame.image.load('data/pyinvader.png')
pygame.display.set_icon(icon)
ufo = pygame.image.load('data/ufo.png')
bg = pygame.image.load('data/bg.png')

# Player
player_img = icon
playerX = width / 2
playerY = 7 * height / 8
playerChange = 5
playerXChange = 0

# Enemy
enemy_img = []
enemyX = []
enemyY = []
enemyChange = 3
enemyXChange = []
enemyYChange = []
no_enemies = 5
for i in range(no_enemies):
    enemy_img.append(ufo)
    enemyX.append(random.randint(0, width - 32))
    enemyY.append(random.randint(0, int(height / 5)))
    enemyXChange.append(enemyChange)
    enemyYChange.append(15)

bullet_img = pygame.image.load('data/bullet.png')

bulletX = 0
bulletY = 7 * height / 8
bulletYChange = 20
bulletState = False

score_value = 0
textX = 10
textY = 10
font = pygame.font.Font('freesansbold.ttf', 16)

gameover = pygame.font.Font('freesansbold.ttf', 32)
clock = pygame.time.Clock()
FPS = 60

# Computer Vision
p_time = 0
c_time = 0
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
x1, x2, xc, y1, y2, yc = [0] * 6
length = 5000
reqlen = 30


def game_over(x, y):
    go = gameover.render("Game Over", True, (255, 255, 255))
    screen.blit(go, (x - 64, y - 32))


def player(x, y):
    screen.blit(player_img, (x, y))


def enemy(x, y):
    screen.blit(ufo, (x, y))


def boundary_player(x, x_change):
    x += x_change

    if x > width - 32:
        x = width - 32
    elif x < 0:
        x = 0

    return x


def boundary_enemy(x, y, x_change, y_change, i):
    if int(x) > width - 32:
        x_change = (-enemyChange)
        x = width - 31
        y += y_change
    elif int(x) < 0:
        x_change = enemyChange
        y += y_change
        x = 1
    x += x_change
    return x, y, x_change


def fire_bullet(x, y, state):
    state = True
    screen.blit(bullet_img, (x + 8, y))
    y -= bulletYChange
    if y < 0:
        state = False
        y = 7 * height / 8
    return x, y, state


def bullet_hit(i):
    global bulletState, score_value, enemyX, enemyY, bulletY
    if enemyX[i] - 16 <= bulletX <= enemyX[i] + 32:
        if enemyY[i] - 16 <= bulletY <= enemyY[i] + 32:
            bulletY = 7 * height / 8
            bulletState = False
            score_value += 1
            enemyX[i] = random.randint(0, width - 32)
            enemyY[i] = random.randint(0, int(height / 4))


def show_score(x, y):
    score = font.render("Score: " + str(score_value), True, (255, 255, 255))
    screen.blit(score, (x, y))


# Game loop
running = True
while running:
    # Computer Vision
    success, img = cap.read()
    img = detector.findhands(img)

    lmList = detector.findposition(img)

    if len(lmList) != 0:
        playerX = (width - xc)

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 255, 0), cv2.FILLED)
        cv2.line(img, (x2, y2), (x1, y1), (255, 255, 0), 3)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < reqlen:
            cv2.circle(img, (xc, yc), 15, (255, 0, 0), cv2.FILLED)
        else:
            cv2.circle(img, (xc, yc), 15, (0, 255, 0), cv2.FILLED)
    else:
        length = 5000
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    img = cv2.flip(img, 1)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (200, 200, 200), 3)

    cv2.imshow("Webcam", img)

    screen.fill((0, 0, 0))
    screen.blit(bg, (0, 0))
    if bulletState:
        bulletX, bulletY, bulletState = fire_bullet(bulletX, bulletY, bulletState)
    playerX = boundary_player(playerX, playerXChange)
    player(playerX, playerY)
    for i in range(no_enemies):
        if enemyY[i] >= playerY - 64:
            for j in range(no_enemies):
                enemyY[j] = 20000
            game_over(width / 2, height / 2)
            break
        enemyX[i], enemyY[i], enemyXChange[i] = boundary_enemy(enemyX[i], enemyY[i], enemyXChange[i], enemyYChange[i],
                                                               i)
        enemy(enemyX[i], enemyY[i])
        bullet_hit(i)
    show_score(textX, textY)
    pygame.display.update()
    if length < reqlen and not bulletState:
        bulletX = playerX
        bulletX, bulletY, bulletState = fire_bullet(bulletX, bulletY, bulletState)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                playerXChange -= playerChange
            if event.key == pygame.K_RIGHT:
                playerXChange += playerChange
            if event.key == pygame.K_SPACE and not bulletState:
                bulletX = playerX
                bulletX, bulletY, bulletState = fire_bullet(bulletX, bulletY, bulletState)
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                playerXChange = 0
    clock.tick(FPS)
