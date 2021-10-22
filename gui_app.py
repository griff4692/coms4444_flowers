import logging
import os

from remi import App, gui


class FlowerApp(App):
    def __init__(self, *args):
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(FlowerApp, self).__init__(*args, static_file_path={'res': res_path})
        self.logger = logging.getLogger(__name__)

    def compute_key(self, row, col):
        if col == 0 and row == 0:
            return 'empty'
        elif col == 0:
            return f'give_{row}'
        elif row == 0:
            return f'give_{col}'
        return f'{row}_{col}'

    def make_bouquet_grid(self, flowers):
        bouquet_grid = gui.GridBox(width='100%', height='100%', style={
            'margin': '0px auto',
            'grid-template-columns': 'repeat(4, 1fr)',
            'grid-template-rows': 'repeat(3, 1fr)',
            'text-align': 'center',
        })
        bouquet_grid.define_grid([[f'{i}_{j}' for j in range(4)] for i in range(3)])
        bouquet_grid.set_column_gap('0%')
        bouquet_grid.set_row_gap('0%')

        if len(flowers) == 0:
            container = gui.Container()
            image = gui.Image(f'/res:coals.png', width='100%', height='100%')
            container.append(image)
            bouquet_grid.append(container, key=f'0_0')
            return bouquet_grid

        flower_idx = 0
        is_done = False
        for i in range(3):
            for j in range(4):
                container = gui.Container(width='100%', height='100%', position='relative')
                container.set_style({
                    'background-color': str(flowers[flower_idx].color.name).lower(),
                    # 'position': 'relative',
                })
                flower_type = str(flowers[flower_idx].type.name).lower()
                sizes = {
                    'small': '33%',
                    'medium': '66%',
                    'large': '100%'
                }
                margins = {
                    'small': '33%',
                    'medium': '16.5%',
                    'large': '0%'
                }
                size = sizes[flowers[flower_idx].size.name.lower()]
                margin = margins[flowers[flower_idx].size.name.lower()]
                image = gui.Image(f'/res:{flower_type}.png')
                image.set_style({
                    'max-height': size,
                    'max-width': size,
                    'width': 'auto',
                    'height': 'auto',
                    'margin': margin
                })
                container.append(image)
                bouquet_grid.append(container, key=f'{i}_{j}')
                flower_idx += 1
                if flower_idx >= len(flowers):
                    is_done = True
                    break
            if is_done:
                break
        return bouquet_grid

    def make_label(self, row, col):
        style = {
            'padding-block-start': '1em',
            'padding-block-end': '1em',
            'margin-block-start': '0em',
            'margin-block-end': '0em',
        }
        name_style = {
            'color': 'purple',
            'font-size': 'large',
            'font-weight': 'bold',
            'border-style': 'inset'
        }

        if col == 0 and row == 0:
            style['background-color'] = 'white'
            label = gui.Label('', style=style)
            return label
        elif col == 0:
            val = f'{self.flower_game.suitors[row - 1].name}_{row - 1} offers...'
            style.update(name_style)
        elif row == 0:
            val = f'{self.flower_game.suitors[col - 1].name}_{col - 1}'
            style.update(name_style)
        else:
            val = 'x'
            grid_style = {
                'color': 'white',
                'font-size': 'large',
                'font-weight': 'bold',
            }
            style.update(grid_style)
        label = gui.Label(val, style=style)
        return label

    def main(self, *userdata):
        self.flower_game,  = userdata
        self.flower_game.set_app(self)

        main_container = gui.Container(style={'width': '100%', 'display': 'block', 'overflow': 'auto', 'text-align': 'center'})

        self.rounds = [f'Round {i}' for i in range(self.flower_game.d)]
        round_drop_down = gui.DropDown.new_from_list(self.rounds, style={'padding': '20px', 'text-align': 'center',})
        round_drop_down.select_by_value(f'Round 0')
        round_drop_down.onchange.do(self.drop_down_changed)

        view_drop_down = gui.DropDown.new_from_list(
            ['Score', 'Rank', 'Bouquet', 'Unions'], style={'padding': '20px', 'text-align': 'center',})
        view_drop_down.select_by_value('Rank')
        view_drop_down.onchange.do(self.view_drop_down_changed)
        self.selected_round = 0
        self.view_val = 'rank'
        self.buttons = {
            'all': gui.Button('Play All', style={'padding': '20px'}),
            'turn': gui.Button('Play Turn', style={'padding': '20px'}),
            'reset': gui.Button('Reset', style={'padding': '20px'}),
            'round': round_drop_down,
            'view': view_drop_down,
        }

        self.buttons['all'].onclick.do(self.simulate_to_end)
        self.buttons['reset'].onclick.do(self.reset)
        self.buttons['turn'].onclick.do(self.simulate_round)

        button_grid = gui.GridBox(width='100%', height='100%', style={
            'margin': '0px auto',
            'grid-template-columns': f'repeat({len(self.buttons)}, 1fr)',
            'grid-template-rows': '1',
            'text-align': 'center',
        })

        for button in self.buttons.values():
            button_grid.append(button)

        n = self.flower_game.p + 1
        self.grid = gui.GridBox(width='100%', height='100%', style={
            'margin': '3px auto',
            'grid-template-columns': f'repeat({n}, 1fr)',
            'grid-template-rows': f'repeat({n}, 1fr)',
            'text-align': 'center',
            'background-color': 'lightslategray'
        })
        self.grid.define_grid([[self.compute_key(i, j) for j in range(n)] for i in range(n)])
        self.grid.set_column_gap('0%')
        self.grid.set_row_gap('0%')
        [[self.grid.append(self.make_label(i, j), key=f'{i}_{j}') for j in range(n)] for i in range(n)]
        main_container.append(button_grid, key='buttons')
        main_container.append(self.grid, key='grid')

        num_marriages = self.flower_game.p // 2
        marriage_grid = gui.GridBox(width='100%', height='100%', style={
            'margin': '0px auto',
            'grid-template-columns': f'repeat({num_marriages}, 1fr)',
            'grid-template-rows': '1',
            'text-align': 'center',
        })

        marriage_grid.define_grid([str(i) for i in range(self.flower_game.p // 2)])
        marriage_grid.set_column_gap('0%')
        marriage_grid.set_row_gap('0%')
        self.marriages = [gui.Label('') for _ in range(num_marriages)]
        [marriage_grid.append(marriage) for marriage in self.marriages]
        main_container.append(marriage_grid)

        return main_container

    def view_drop_down_changed(self, widget, value):
        if (value.lower() == 'unions' and not self.flower_game.is_over()) or self.flower_game.next_round == 0:
            self.buttons['view'].select_by_value(self.view_val.capitalize())
        else:
            self.view_val = value.lower()
            if self.view_val == 'unions':
                self.selected_round = self.flower_game.d - 1
                self.buttons['round'].select_by_value(self.rounds[-1])
            self.update_table_by_round()

    def drop_down_changed(self, widget, value):
        if self.view_val == 'unions':
            self.buttons['round'].select_by_value(self.rounds[-1])
        else:
            round_selected = int(value.split(' ')[-1])
            if round_selected > self.flower_game.next_round - 1:
                self.selected_round = max(0, self.flower_game.next_round - 1)
                self.buttons['round'].select_by_value(f'Round {self.selected_round}')
            else:
                self.selected_round = round_selected
            self.update_table_by_round()

    def reset(self, widget):
        self.flower_game.reset_game_state()
        self.selected_round = 0
        self.buttons['round'].select_by_value('Round 0')
        self.buttons['view'].select_by_value('Rank')
        self.view_val = 'rank'
        [self.marriages[i].set_text('') for i in range(len(self.marriages))]
        [self.marriages[i].set_style({'border': 'none'}) for i in range(len(self.marriages))]
        self.update_table_by_round()

    def update_table_by_bouquet_round(self):
        round_bouquets = self.flower_game.bouquets[self.selected_round]
        for i in range(self.flower_game.p):
            for j in range(self.flower_game.p):
                if i == j:
                    continue
                bouquet = round_bouquets[i, j]
                flowers = bouquet.flowers()
                self.grid.children[f'{i + 1}_{j + 1}'] = self.make_bouquet_grid(flowers)

    def update_table_by_round(self):
        if self.view_val == 'unions':
            self.update_w_marriages(self.flower_game.marriages)
            return
        if self.view_val == 'bouquet':
            self.update_table_by_bouquet_round()
            return

        val_mat = (self.flower_game.ranks[self.selected_round]
                   if self.view_val == 'rank' else self.flower_game.scores[self.selected_round])
        for i in range(self.flower_game.p):
            for j in range(self.flower_game.p):
                if type(self.grid.children[f'{i + 1}_{j + 1}']) == gui.GridBox:
                    self.grid.children[f'{i + 1}_{j + 1}'] = self.make_label(i + 1, j + 1)
                if i == j:
                    style = {'background-color': 'rgb(210, 210, 210, 0.25)'}
                else:
                    raw_score = val_mat[i, j]
                    if self.view_val == 'rank':
                        max_val = val_mat.max()
                        min_val = 1
                        scaled_val = 1 - (raw_score - min_val) / (max_val - min_val)
                    else:
                        scaled_val = val_mat[i, j]
                        raw_score = round(raw_score, 2)
                    style = {'background-color': f'rgb(255, 0, 0, {max(0.1, scaled_val)})', 'font-weight': 'bold'}
                    key = str(raw_score)
                    self.grid.children[f'{i + 1}_{j + 1}'].set_text(key)
                self.grid.children[f'{i + 1}_{j + 1}'].set_style(style)

    def simulate_round(self, widget):
        if self.flower_game.is_over():
            self.logger.warning('Already finished!  Cannot simulate anymore round')
        else:
            self.flower_game.simulate_next_round()
            self.update()

    def update_w_marriages(self, marriage_outputs):
        for i in range(self.flower_game.p):
            for j in range(self.flower_game.p):
                if type(self.grid.children[f'{i + 1}_{j + 1}']) == gui.GridBox:
                    self.grid.children[f'{i + 1}_{j + 1}'] = self.make_label(i + 1, j + 1)
                if i == j:
                    continue
                else:
                    adv_val = round(self.flower_game.advantage[i][j], 2)
                    self.grid.children[f'{i + 1}_{j + 1}'].set_text(f'{adv_val}')
                    style = {'background-color': 'rgb(225,225,225,0.5)', 'font-weight': 'light'}
                    self.grid.children[f'{i + 1}_{j + 1}'].set_style(style)
        for marriage_order, pair in enumerate(marriage_outputs['unions']):
            opacity = (len(marriage_outputs['unions']) - marriage_order) / len(marriage_outputs['unions'])
            style = {'background-color': f'rgb(255,0,0,{opacity})', 'font-weight': 'bold'}
            raw_score = marriage_outputs['scores'][pair['chooser']]
            score = round(raw_score, 2)
            suitor_id = pair['suitor']
            chooser_id = pair['chooser']
            self.grid.children[f'{suitor_id + 1}_{chooser_id + 1}'].set_text(str(score))
            self.grid.children[f'{suitor_id + 1}_{chooser_id + 1}'].set_style(style)
            suitor_name = f'{self.flower_game.suitors[suitor_id].name}_{suitor_id}'
            chooser_name = f'{self.flower_game.suitors[chooser_id].name}_{chooser_id}'
            proposal_text = f'{chooser_name} accepted {suitor_name}\'s proposal on January {marriage_order + 1}'
            self.marriages[marriage_order].set_text(proposal_text)
            union_style = {'border-top': 'solid', 'border-bottom': 'solid', 'border-left': 'solid', 'padding': '20px'}
            if marriage_order == len(marriage_outputs['unions']) - 1:
                union_style['border-right'] = 'solid'
            self.marriages[marriage_order].set_style(union_style)

    def update(self):
        if self.flower_game.is_over() and self.flower_game.marriages is not None:
            self.buttons['round'].select_by_value(self.rounds[-1])
            self.selected_round = self.flower_game.d - 1
            self.view_val = 'unions'
            self.buttons['view'].select_by_value('Unions')
            self.update_w_marriages(self.flower_game.marriages)
        else:
            curr_round = self.flower_game.next_round - 1
            self.buttons['round'].select_by_value(f'Round {curr_round}')
            self.selected_round = curr_round
            self.update_table_by_round()

    def simulate_to_end(self, widget):
        if self.flower_game.is_over():
            self.logger.warning('Already finished!  Cannot simulate anymore round')
        else:
            self.flower_game.play()
            self.update()
