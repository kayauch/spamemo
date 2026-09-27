"""Generic retrieval hints, not observed properties or stored location evidence."""
import unicodedata

_GROUPS = [
    (("はさみ", "ハサミ", "鋏", "scissors"), "scissors はさみ。紙や布を切る道具。"),
    (("カッター", "カッターナイフ", "utility knife"), "utility knife カッターナイフ。紙や段ボールを切る刃物。"),
    (("充電ケーブル", "usbケーブル", "usbコード", "charging cable"), "charging cable USB cable 充電ケーブル。スマホなどの機器を充電するコード。"),
    (("充電器", "acアダプター", "charger"), "charger 充電器。コンセントから機器に電力を供給するアダプター。"),
    (("眼鏡", "メガネ", "めがね", "glasses"), "eyeglasses 眼鏡。文字や物を見るために目にかけるもの。"),
    (("ホッチキス", "ステープラー", "stapler"), "stapler ホッチキス。紙や書類を針でとじる道具。"),
    (("セロハンテープ", "セロテープ", "粘着テープ"), "adhesive tape 粘着テープ。紙や物を貼り付けるもの。"),
    (("ドライバー", "ねじ回し", "screwdriver"), "screwdriver ドライバー。ねじを締めたり緩めたりする工具。"),
    (("定規", "ものさし", "ruler"), "ruler 定規。長さを測ったり直線を引いたりする道具。"),
    (("リモコン", "remote control"), "remote control リモコン。家電を離れた場所から操作するもの。"),
]


def retrieval_hint(entity):
    # Whole identity only: do not infer attributes from free-form notes or substrings.
    terms = {unicodedata.normalize('NFKC', s).lower().strip()
             for s in [entity.name, *entity.aliases]}
    return ' '.join(hint for aliases, hint in _GROUPS if terms.intersection(aliases))
