class StableDiffusionModelHijack:

    def apply_optimizations(self, option=None):
        pass

    def convert_sdxl_to_ssd(self, m):
        pass

    def hijack(self, m):
        pass

    def undo_hijack(self, m):
        pass

    def apply_circular(self, enable):
        pass

    def clear_comments(self):
        pass

    def get_prompt_lengths(self, text, cond_stage_model):
        from modules import shared
        return shared.sd_model.get_prompt_lengths_on_ui(text)

    def redo_hijack(self, m):
        pass


model_hijack = StableDiffusionModelHijack()
