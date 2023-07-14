"""
Main Driver of the Project: This code is designed using Pygame which is based on the concept of Canvas drawing

Stage 1: An image is picked from the test data set and passed to the predictor model

Stage 2: Segmented Image is then constructed on Pygame based on step size [ Segmentation is translated
using contours drawn over the segmentation with the help of openCV - These form the initial set of obstacles

Stage 3: User Position and Destination needs to be marked by initial mouse clicks

Stage 4: Path planning algorithm is called using these information and the path is drawn

Stage 5: A scanner [Rectangular box in front of the robot] is attached to the bot icon

Stage 6: As the scanner moves ahead, that section of image is captured and passed to the Yolo object detector

Stage 7: If the obstacle is detected in that slice, one position ahead of the robot and one at end of the obstacle
            is used to construct an alternative shortest path

        If an important object is detected, position co-ordinates are saved for later use
"""

import cv2
import pygame
from scipy.spatial import distance

from configuration import Config as config
from object_detection.yolo_detector import Detector
from path_planning.astar import find_shortest_path
from predict import predict_obstacles
from utils.obstacle_type import Obstacle

# initialisations
user_pos_x = None
user_pos_y = None
user_locked = False

target_pos_x = None
target_pos_y = None
target_locked = False

user = None
user_guide = None
step = 0

clock = pygame.time.Clock()
detector = Detector()

fps = 60

# Canvas initialisation
background_colour = (255, 255, 255)
(width, height) = (1024, 768)
obstacle_color = (0, 0, 0)
user_color = (0, 0, 255)
target_color = (0, 27, 100)
guide_marker_color = (255, 0, 0)

user_icon = pygame.image.load(f'{config.asset_path}user.png')

path_drawn = False
path_list = []

obstacle_co_ordinates = set()
imp_list = set()

master_path_list = []
coll_list = []

dog = pygame.image.load(f'{config.asset_path}dog_icon.png')
bottle = pygame.image.load(f'{config.asset_path}bottle.png')

obstacles_pos = []
imp_pos = []


def convert_to_tens(x_pos, y_pos):
    """
    Utility function which converts each pixel position of an image to a factor of the chosen step size
    :param x_pos: x co-ordinate
    :param y_pos: y co-ordinate
    :return:
    """
    vx = x_pos % config.step_size
    vy = y_pos % config.step_size
    if vx > 5:
        x_pos += config.step_size - vx
    else:
        x_pos -= vx
    if vy > 5:
        y_pos += config.step_size - vy
    else:
        y_pos -= vy
    return x_pos, y_pos


def draw_icon(icon, pos_x, pos_y):
    """
    Draw icons on the screen
    :param icon: image
    :param pos_x: pos x
    :param pos_y: pos y
    """
    screen.blit(icon, (pos_x, pos_y))


if __name__ == '__main__':
    # Read a floor plan
    img_path = f'{config.test_folder_images}base_12.png'
    # Segment a floor plan
    floor_plan = predict_obstacles(img_path)
    # Issue: Need to write it locally and read again
    cv2.imwrite(f'{config.output_path}prediction.png', floor_plan)

    floor_plan = cv2.imread(f'{config.output_path}prediction.png')
    floor_plan = cv2.cvtColor(floor_plan, cv2.COLOR_BGR2GRAY)
    floor_plan = floor_plan * 255

    # Useful to get the contours clearly
    ret, thresh = cv2.threshold(floor_plan, 150, 255, cv2.THRESH_BINARY)

    ctr, _ = cv2.findContours(thresh,
                              cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    list_ctr = []
    for cont in ctr:
        list_ctr.append(cv2.boundingRect(cont))

    # Start of pygame
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Smarter-Vision Assist Demo')
    screen.fill(background_colour)

    for val in list_ctr:
        (x, y, w, h) = val
        # print(val)
        for v1 in range(x, x + w, config.step_size):
            for v2 in range(y, y + h, config.step_size):
                v1, v2 = convert_to_tens(v1, v2)
                obj = Obstacle(pygame.Rect(v1, v2, config.step_size, config.step_size), 0)
                pygame.draw.rect(screen, obstacle_color, obj.rectangle)
                obstacle_co_ordinates.add(obj)

    pygame.display.flip()
    stage_one = True
    is_path_drawn = False

    while stage_one:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stage_one = False

            if event.type == pygame.MOUSEBUTTONUP:
                if user_pos_x is None:
                    user_pos_x, user_pos_y = pygame.mouse.get_pos()

                if user_pos_x is not None and target_pos_x is None and user_locked:
                    target_pos_x, target_pos_y = pygame.mouse.get_pos()

        # draw target
        if target_pos_x is not None and target_pos_y is not None:
            target_pos_x, target_pos_y = convert_to_tens(target_pos_x, target_pos_y)
            pygame.draw.circle(screen, target_color, (target_pos_x, target_pos_y), config.step_size)
            pygame.display.update()
            target_locked = True

        # draw user
        if user_pos_x is not None and user_pos_y is not None:
            user_pos_x, user_pos_y = convert_to_tens(user_pos_x, user_pos_y)
            pygame.draw.circle(screen, user_color, (user_pos_x, user_pos_y), config.step_size)
            user_guider = pygame.draw.line(screen, guide_marker_color, (user_pos_x, user_pos_y),
                                           (user_pos_x + config.step_size, user_pos_y), 2)
            pygame.display.update()
            user_locked = True

        # get initial shortest path
        if not path_drawn and user_locked and target_locked:
            path_list = find_shortest_path((user_pos_x, user_pos_y), (target_pos_x, target_pos_y),
                                           obstacle_co_ordinates)
            master_path_list.extend(path_list)
            path_drawn = True

        # draw initial shortest path
        if len(master_path_list) > 0:
            for i in range(0, len(master_path_list) - 1):
                line_rect = pygame.draw.line(screen, [0, 255, 0], master_path_list[i], master_path_list[i + 1], 5)
                coll_list.append(line_rect)
                pygame.display.update()
                clock.tick(fps)
            is_path_drawn = True
            stage_one = False

    # stage 2 initialisation
    second_stage = True

    user_angle = 0

    obstacle_x = None
    obstacle_y = None

    has_collided = False

    # step controller
    d = 0

    # enablers
    isShift = False
    isCtrl = False

    # refer event.key code for Left Shift and Right Shift values
    left_shift = 1073742049
    right_shift = 1073742053

    tile_width = 0
    tile_height = 0

    colliding_obj = None

    while second_stage and not stage_one:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stage_one = False

            elif event.type == pygame.KEYUP:
                if event.key == left_shift:
                    isShift = True
                    isCtrl = False
                elif event.key == right_shift:
                    isCtrl = True
                    isShift = False

            if event.type == pygame.MOUSEBUTTONUP:
                obstacle_x, obstacle_y = pygame.mouse.get_pos()
                ob = pygame.rect.Rect(obstacle_x - 10, obstacle_y - 25, 100, 100)
                if isShift:
                    obstacles_pos.append(ob)
                    isShift = False
                elif isCtrl:
                    imp_pos.append(ob)
                    isCtrl = False

        # draw path
        if master_path_list is not None:
            for i in range(0, len(master_path_list) - 1):
                pygame.draw.line(screen, [0, 255, 0], master_path_list[i], master_path_list[i + 1], 5)

        # draw obstacle - dog
        if len(obstacles_pos):
            for rec in obstacles_pos:
                draw_icon(dog, rec.x, rec.y)

        # draw important items
        if len(imp_pos):
            for rec in imp_pos:
                draw_icon(bottle, rec.x, rec.y)

        # Mark or fill obstacles
        for obj in obstacle_co_ordinates:
            if obj.mode == 0:
                pygame.draw.rect(screen, obstacle_color, obj.rectangle)
            elif obj.mode == 1:
                pygame.draw.rect(screen, (255, 0, 0), obj.rectangle, 2)
            elif obj.mode == 2:
                pygame.draw.rect(screen, (0, 255, 0), obj.rectangle, 2)

        if len(obstacle_co_ordinates) > 0:

            # check if the user is nearing any of the obstacles

            for obj in obstacle_co_ordinates:
                obs = obj.rectangle
                val = obs.collidelist(coll_list)
                if val != -1:
                    val = coll_list[val]
                    if user_pos_x <= val.x and user_pos_y >= val.y and obj.mode != 0:
                        collision_dist = distance.euclidean((user_pos_x, user_pos_y), (val.x, val.y))
                        print(f'Collision Distance to an Obstacle {collision_dist}')
                        if 60 >= collision_dist >= 19 and not has_collided:
                            # Random tested values for thresholding nearness
                            d = int(((val.x + obs.w) - user_pos_x) / config.step_size)
                            colliding_obj = obj
                            has_collided = True

        # draw destination
        pygame.draw.circle(screen, target_color, (target_pos_x, target_pos_y), config.step_size)

        # draw moving bot/user
        draw_icon(user_icon,
                  user_pos_x - (int(user_icon.get_width() / 2)),
                  user_pos_y - (int(user_icon.get_height() / 2)))

        """
        For every step check the following:
        1. Is user approaching any obstacles
        2. If yes, find an alternative route else proceed
        3. Place a scanner (rectangle) to scan obstacles ahead of user
        4. Send scanned images for objection detection model
        """
        if step < len(master_path_list):
            prev_x = user_pos_x
            prev_y = user_pos_y
            user_pos_x, user_pos_y = master_path_list[step][0], master_path_list[step][1]
            user_icon_copy = user_icon

            if has_collided:
                new_x, new_y = master_path_list[step + 1][0], master_path_list[step + 1][1]
                if d == 0:
                    d = d + 1
                dst_x, dst_y = master_path_list[step + d + 2][0], master_path_list[step + d + 2][1]

                new_path = find_shortest_path((new_x, new_y), (dst_x, dst_y), obstacle_co_ordinates)

                x_path = master_path_list[:step + 1]
                y_path = master_path_list[step + d + 2:]
                x_path.extend(new_path)
                x_path.extend(y_path)

                master_path_list = x_path
                d = 0
                if colliding_obj is not None:
                    obstacle_co_ordinates.remove(colliding_obj)

            if prev_x != user_pos_x:
                xx = prev_x + 13
                yy = prev_y - 58
                tile_width = 80
                tile_height = 100
            else:
                xx = prev_x - 45
                yy = prev_y - 95
                tile_width = 100
                tile_height = 80

            # capture scanned portion
            scanned_portion = pygame.draw.rect(screen, (255, 0, 0), pygame.rect.Rect(xx, yy, tile_width, tile_height),
                                               1)
            sub = screen.subsurface(scanned_portion)
            sub_arr = pygame.surfarray.array3d(sub)

            # send scanned portion for object detection
            image_dict = detector.detect_object(sub_arr)

            # segregate obstacle and important object and add to list
            if image_dict.get('obstacles') is not None:
                for val in image_dict.get('obstacles'):
                    val.x = xx + val.x - 10
                    val.y = yy + val.y
                    val.h += 10
                    val.w += 10
                    obj = Obstacle(val, 1)
                    obstacle_co_ordinates.add(obj)

            if image_dict.get('important') is not None:
                for val in image_dict.get('important'):
                    val.x = xx + val.x - 10
                    val.y = yy + val.y
                    val.h += 10
                    val.w += 10
                    obj = Obstacle(val, 2)
                    obstacle_co_ordinates.add(obj)

            if prev_x == user_pos_x:
                # move up, facing up

                if user_angle == 90:
                    user_icon_copy = pygame.transform.rotate(user_icon, 90)
                    draw_icon(user_icon_copy,
                              prev_x - (int(user_icon_copy.get_width() / 2)),
                              prev_y - (int(user_icon_copy.get_height() / 2)))
                    pygame.display.update()
                else:

                    # moving up, facing down
                    for val in range(user_angle, 90):
                        user_icon_copy = pygame.transform.rotate(user_icon, val)

                        draw_icon(user_icon_copy,
                                  prev_x - (int(user_icon_copy.get_width() / 2)),
                                  prev_y - (int(user_icon_copy.get_height() / 2)))

                        pygame.display.update()
                        pygame.time.delay(10)
                    user_angle = 90
            else:
                # moving right
                user_icon_copy = pygame.transform.rotate(user_icon, 0)
                user_angle = 0
                draw_icon(user_icon_copy,
                          prev_x - (int(user_icon_copy.get_width() / 2)),
                          prev_y - (int(user_icon_copy.get_height() / 2)))
            step += 1
        else:
            stopper = True

        pygame.display.update()
        pygame.time.delay(1000)
        has_collided = False
