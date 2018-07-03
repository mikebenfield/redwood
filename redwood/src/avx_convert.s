.intel_syntax noprefix
.text
	.globl _redwood_avx2_to_f16_8, redwood_avx2_to_f16_8

	// win64 calling convention: rcx is dest; rdx is source
_redwood_avx2_to_f16_8:
redwood_avx2_to_f16_8:
	vmovups		ymm0, [rdx]		// input(ymm0): f32
	vcvtps2ph	xmm1, ymm0, 0		// let as_i16(xmm1): i32 = transmute(to_f16(input))
	mov		rax, 0x7FFF7FFF
	vmovd		xmm2, rax		// set up xmm2 = 0x7FFF
	vpbroadcastw	xmm2, xmm2
	vpxor		xmm4, xmm4, xmm4	// set up xmm4 = 0
	vpand		xmm3, xmm1, xmm2	// let non_sign_bits(xmm3) = as_i16 & 0x7FFF;
	vpcmpgtw	xmm5, xmm4, xmm1	// let sign(xmm5): i16 = if 0 > as_i16 { -1 } else { 0 };
	vpmullw		xmm6, xmm3, xmm5	// let result_if_negative(xmm6) = non_sign_bits * sign;
	vpandn		xmm7, xmm5, xmm1	// let result_if_nonnegative(xmm7) = as_i16 & (!sign);
	vpor		xmm3, xmm6, xmm7	// return(xmm1) result_if_negative  result_if_nonnegative;
	vmovups		[rcx], xmm3
	ret


	.globl _redwood_avx2_to_f32_8, redwood_avx2_to_f32_8

	// win64 calling convention: rcx is dest; rdx is source
_redwood_avx2_to_f32_8:
redwood_avx2_to_f32_8:
	vmovups		xmm0, [rdx]
	vpxor		xmm1, xmm1, xmm1
	mov		rax, 0x7FFF7FFF
	vmovd		xmm2, rax
	vpand		xmm3, xmm2, xmm0
	vpcmpgtw	xmm5, xmm1, xmm0
	vpmullw		xmm6, xmm3, xmm5
	vpandn		xmm7, xmm5, xmm0
	vpor		xmm8, xmm6, xmm7
	vcvtph2ps	ymm9, xmm8
	vmovups		[rcx], ymm9
	ret
